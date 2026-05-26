import ssl
import typer
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from transformers import (T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          AutoModelForCausalLM, AutoTokenizer)
from datasets import load_dataset
import wandb
from pathlib import Path
import shutil

from ConvFQ import ConvFQ
from engine import train_epoch, evaluate

from models.SmallNet import SmallNet
from models.vit import ViTWrapper
from fq import frankensteinize
from gumbel_bit_quantizer import ModuleQuantizer
from computer_analyser import ComputeAnalyser

ssl._create_default_https_context = ssl._create_unverified_context
app = typer.Typer()

BIT_CHOICES = [2, 4, 8, 16]
COST_TABLE = {2: 0.5, 4: 1.0, 8: 2.0, 16: 3.0}

class OpusBooks(Dataset):
    
    def __init__(self, tokenizer, debug, dataset, num_debug_samples):
        self.dataset, self.tokenizer = dataset.select(list(range(0, num_debug_samples))) if debug else dataset, tokenizer

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        source = self.dataset[index]['translation']['en']
        target = self.dataset[index]['translation']['fr']
        source = self.tokenizer([source], max_length=512, padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer([target], max_length=512, padding='max_length', truncation=True, return_tensors="pt")

        return {"source_ids": source['input_ids'].squeeze(), "source_mask": source['attention_mask'].squeeze(), "target_ids": targets['input_ids'].squeeze(), "target_mask": targets['attention_mask'].squeeze()}

class DolciDataset(Dataset):
    """Dataset for instruction-following with AllenAI Dolci-Instruct-SFT dataset."""

    def __init__(self, tokenizer, debug, dataset, num_debug_samples=10000, max_length=512):
        """
        Initialize Dolci dataset.

        Args:
            tokenizer: Causal LM tokenizer
            debug: If True, use subset for debugging
            dataset: HuggingFace dataset with instruction/response pairs
            num_debug_samples: Number of samples to use in debug mode
            max_length: Maximum sequence length
        """
        self.dataset = dataset.select(list(range(0, min(num_debug_samples, len(dataset))))) if debug else dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Debug: print first example to understand format
        if len(self.dataset) > 0:
            print(f"Dataset format (first example): {self.dataset[0].keys()}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Get a single training example.

        AllenAI Dolci-Instruct-SFT dataset format:
        - 'messages': List of message objects with 'role' and 'content' fields
        - 'source_dataset': Original dataset source
        - 'domain': Domain category
        """
        example = self.dataset[index]

        # Handle AllenAI Dolci-Instruct-SFT format with messages
        if 'messages' in example:
            messages = example['messages']

            # Extract instruction and response from messages
            if isinstance(messages, list) and len(messages) >= 2:
                # Messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
                user_messages = []
                assistant_messages = []

                for msg in messages:
                    role = msg.get('role', '')
                    content = msg.get('content', '')

                    if role == 'user':
                        user_messages.append(content)
                    elif role == 'assistant':
                        assistant_messages.append(content)

                # Concatenate all user messages as instruction, last assistant as response
                instruction = ' '.join(user_messages) if user_messages else "Follow the instructions."
                response = assistant_messages[-1] if assistant_messages else "I understand."

                # Handle edge cases where content might be empty
                if not instruction.strip():
                    instruction = "Please provide a response."
                if not response.strip():
                    response = "I'll help you with that."

            else:
                raise ValueError(f"Unexpected messages format at index {index}. Messages: {messages}")

        # Handle simpler instruction/output formats
        elif 'instruction' in example and 'output' in example:
            instruction = example['instruction']
            response = example['output']
        elif 'instruction' in example and 'response' in example:
            instruction = example['instruction']
            response = example['response']
        elif 'prompt' in example and 'completion' in example:
            instruction = example['prompt']
            response = example['completion']
        elif 'input' in example and 'output' in example:
            instruction = example['input']
            response = example['output']
        else:
            # Fallback: try to detect text fields automatically
            text_fields = []
            for k, v in example.items():
                if isinstance(v, str) and len(v) > 10:  # Only substantial text fields
                    text_fields.append(k)

            if len(text_fields) >= 2:
                instruction = example[text_fields[0]]
                response = example[text_fields[1]]
            else:
                raise ValueError(f"Unsupported dataset format at index {index}. Available keys: {example.keys()}")

        # Format for causal LM: instruction + response
        text = f"Instruction: {instruction}\nResponse: {response}"

        # Tokenize with causal LM format
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # For causal LM, labels are the same as input_ids (shifted during training)
        return {
            "input_ids": encoded['input_ids'].squeeze(),
            "attention_mask": encoded['attention_mask'].squeeze(),
            "labels": encoded['input_ids'].squeeze()  # Labels are same as input for causal LM
        }

def save_checkpoint(model, optimizer, epoch, loss, acc, test_loss, test_acc, args, quantization_metadata=None):
    """
    Save model checkpoint with all necessary information for later evaluation.

    Args:
        model: The trained model
        optimizer: The optimizer state
        epoch: Current epoch number
        loss: Training loss
        acc: Training accuracy
        test_loss: Test loss
        test_acc: Test accuracy
        args: All training arguments for reproducibility
        quantization_metadata: Dictionary with quantization info (bit choices, alphas, etc.)
    """
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Create experiment-specific directory name
    model_type = args.get('model_type', 'unknown') if isinstance(args, dict) else getattr(args, 'model_type', 'unknown')
    bit_choices = args.get('bit_choices', []) if isinstance(args, dict) else getattr(args, 'bit_choices', [])
    lambda_cost = args.get('lambda_cost', 0.0) if isinstance(args, dict) else getattr(args, 'lambda_cost', 0.0)

    exp_name = f"{model_type}_epochs{epoch}_lambda{lambda_cost}_bits{str(bit_choices)}"
    exp_dir = checkpoint_dir / exp_name
    exp_dir.mkdir(exist_ok=True)

    print(f"Saving checkpoint to {exp_dir}...")

    # Save model state
    model_path = exp_dir / "model.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'acc': acc,
        'test_loss': test_loss,
        'test_acc': test_acc,
    }, model_path)

    # Save quantization metadata
    if quantization_metadata:
        metadata_path = exp_dir / "quantization_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(quantization_metadata, f, indent=2)

    # Save training arguments for reproducibility
    args_path = exp_dir / "training_args.json"
    # Convert args to dict if it's a Namespace object
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    else:
        args_dict = args

    # Filter out non-serializable items
    serializable_args = {}
    for key, value in args_dict.items():
        try:
            json.dumps(value)  # Test if serializable
            serializable_args[key] = value
        except (TypeError, ValueError):
            serializable_args[key] = str(value)  # Convert to string if not serializable

    with open(args_path, 'w') as f:
        json.dump(serializable_args, f, indent=2)

    # Save final bit-widths for each quantized layer
    if quantization_metadata and 'layer_bit_widths' in quantization_metadata:
        bits_path = exp_dir / "final_bit_widths.json"
        with open(bits_path, 'w') as f:
            json.dump(quantization_metadata['layer_bit_widths'], f, indent=2)

    print(f"✓ Checkpoint saved: {exp_dir}")
    print(f"  - Model: {model_path}")
    print(f"  - Metadata: {metadata_path if quantization_metadata else 'N/A'}")
    print(f"  - Args: {args_path}")

    return str(exp_dir)

def extract_quantization_metadata(model):
    """
    Extract quantization metadata from a trained model.

    Args:
        model: Trained model with ModuleQuantizer instances

    Returns:
        Dictionary containing quantization metadata
    """
    metadata = {
        'layer_bit_widths': {},
        'layer_alpha_params': {},
        'quantized_layers': [],
        'cost_table': {},
        'bit_choices': []
    }

    for name, module in model.named_modules():
        if isinstance(module, ModuleQuantizer):
            metadata['quantized_layers'].append(name)

            # Save final bit choice
            if hasattr(module, 'chosen_bit') and module.chosen_bit is not None:
                metadata['layer_bit_widths'][name] = module.chosen_bit
            elif hasattr(module, 'alpha'):
                # If not finalized, save the current best choice
                idx = module.alpha.argmax().item()
                chosen_bit = module.bit_choices[idx]
                metadata['layer_bit_widths'][name] = chosen_bit

            # Save alpha parameters for reconstruction
            if hasattr(module, 'alpha'):
                metadata['layer_alpha_params'][name] = {
                    'alpha_values': module.alpha.detach().cpu().tolist(),
                    'bit_choices': module.bit_choices
                }

            # Save cost table and bit choices (should be same for all layers)
            if hasattr(module, 'cost_table') and not metadata['cost_table']:
                metadata['cost_table'] = {str(k): v for k, v in module.cost_table.items()}
            if hasattr(module, 'bit_choices') and not metadata['bit_choices']:
                metadata['bit_choices'] = module.bit_choices

    return metadata

@app.command()
def main(
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    lambda_cost: float = 0.001,
    alpha_lr_mult: float = 20.0,
    cost_reduction: str = "sum",
    use_quant: bool = False,
    bit_choices: str = None,
    log: bool = False,
    model_type: str = "vit",
    debug: bool = False, # reduce number of sampels for debugging
    true_costs: bool = False, # use measured costs instead of proxy costs
    include_cnn: bool = False,
    seed: int = 42,
    deterministic: bool = True,
    num_workers: int = 0,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reproducibility controls
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    test_generator = torch.Generator()
    test_generator.manual_seed(seed)
    
    print(f"model type: {model_type}")
    if use_quant:
        if bit_choices is not None:     
            bit_choices = eval(bit_choices)
        else:
            print("Using default bit choices [2,4,8,16]")
            bit_choices = BIT_CHOICES

    # Determine cost table
    if true_costs:
        analyser = ComputeAnalyser()
        cost_table_raw = analyser.analyse()
        print("Using true cost table:", cost_table_raw)
        with open("compute_benchmark_results.json", "r") as f:
            data = json.load(f)
            cost_table_benchmark = data[0]['benchmarks']['2048x2048']
        # balance compute and memory costs 
        current_cost_table = {int(k): v['time_ms'] for k, v in cost_table_benchmark.items()}
        print("Using true cost table:", current_cost_table)     
    else:
        current_cost_table = COST_TABLE

    if model_type.lower() in ["t5"]:
        model_id = 't5-small'
        tokenizer =  T5Tokenizer.from_pretrained(model_id, legacy=False)
        train_test_split = load_dataset("Helsinki-NLP/opus_books", "en-fr", split="train").train_test_split(test_size=0.2) # Fetch from Huggingface
        trainset = OpusBooks(tokenizer, debug, train_test_split['train'], num_debug_samples=15000)
        testset = OpusBooks(tokenizer, debug, train_test_split['test'], num_debug_samples=3000)
    elif model_type.lower() in ["decoder"]:
        # OLMo3 or other decoder models
        model_id = 'allenai/Olmo-3-1025-7B'
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Set pad token if it doesn't exist (common for decoder-only models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load AllenAI Dolci dataset (official instruction-following dataset)
        try:
            # Load official AllenAI Dolci dataset
            dolci_dataset = load_dataset("allenai--Dolci-Instruct-SFT", split="train")
            print(f"Loaded AllenAI Dolci-Instruct-SFT dataset: {len(dolci_dataset)} samples")
        except Exception as e:
            print(f"Could not load AllenAI Dolci dataset: {e}")
            print("Trying to load with different configuration...")
            try:
                # Try without split specification
                dolci_dataset = load_dataset("allenai--Dolci-Instruct-SFT")
                if isinstance(dolci_dataset, dict):
                    dolci_dataset = dolci_dataset['train']
                print(f"Loaded AllenAI Dolci-Instruct-SFT dataset: {len(dolci_dataset)} samples")
            except Exception as e2:
                print(f"Still could not load Dolci dataset: {e2}")
                print("Using a subset of OpenAI's webgpt_completions as fallback...")
                # Fallback to a different instruction dataset
                dolci_dataset = load_dataset("openai/webgpt_completions", split="train")

        # Split into train/test
        if len(dolci_dataset) > 11000:
            # Use subset for faster iteration as per plan
            train_dataset = dolci_dataset.select(range(10000))
            test_dataset = dolci_dataset.select(range(10000, 11000))
        else:
            # Use 80/20 split if dataset is smaller
            split_dataset = dolci_dataset.train_test_split(test_size=0.2)
            train_dataset = split_dataset['train']
            test_dataset = split_dataset['test']

        trainset = DolciDataset(tokenizer, debug, train_dataset, num_debug_samples=10000)
        testset = DolciDataset(tokenizer, debug, test_dataset, num_debug_samples=2000)
    else:
        model_id = None
        transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=test_generator,
    )

    if log:
        if model_type.lower() == "decoder":
            run_name = f"{model_type.upper()}_Dolci" + (f"_Quant_{str(bit_choices)}" if use_quant else "_FullPrec")
        else:
            run_name = f"{model_type.upper()}_CIFAR100" + (f"_Quant_{str(bit_choices)}" if use_quant else "_FullPrec")

        project_tags = {
            "vit": f"vit_lambda_{lambda_cost}",
            "smallnet": f"smallnet_lambda_{lambda_cost}",
            "t5": f"t5_lambda_{lambda_cost}",
            "decoder": f"decoder_lambda_{lambda_cost}"
        }

        tag = project_tags.get(model_type.lower(), f"{model_type}_lambda_{lambda_cost}")
        wandb.init(project="scaling-frankenstein-quant-t5", name=run_name+f"_l_{str(lambda_cost)}", tags=tag)
    
    match model_type.lower():
        case "vit":
            typer.echo("Using ViT model")
            model = ViTWrapper(num_classes=100).to(device)
        case "smallnet":
            typer.echo("Using SmallNet model")
            model = SmallNet().to(device)
        case "t5":
            typer.echo("Using T5 model")
            model = T5ForConditionalGeneration(T5Config.from_pretrained(model_id)).to(device)
        case "decoder":
            typer.echo("Using OLMo3-7B decoder model")
            # Load OLMo3 with optimizations for memory management
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
                trust_remote_code=True,
                device_map="auto"  # Automatic device placement
            )
        case _:
            raise ValueError(f"Unknown model type: {model_type}")

    if use_quant:
        typer.echo("Applying Frankenstein Quantization...")
        model = frankensteinize(model, new_class_kwargs={
            "name": "fc",
            # "name": "T5Attention",
            "bit_choices": bit_choices,
            "cost_table": current_cost_table
        })
        if include_cnn:
            typer.echo("Also applying Frankenstein Quantization to Conv2d layers...")
            model = frankensteinize(model, old_class=nn.Conv2d, new_class=ConvFQ, new_class_kwargs={
                "name": "conv2d",
                "bit_choices": bit_choices,
                "cost_table": current_cost_table
            })

    print("\n=== Model Summary ===")
    print(model)
    model.to(device)

    if log:
         wandb.config.update({
            "epochs": 10,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "seed": seed,
            "deterministic": deterministic,
            "bit_choices": bit_choices,
            "cost_table": current_cost_table,
            "use_quant": use_quant,
            "cost_reduction": cost_reduction,
            "model": str(model)
        })
    if use_quant:
        alpha_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".alpha"): # learnable logits in ModuleQuantizer should have separate LR
                alpha_params.append(param)
            else:
                other_params.append(param)
        
        print("\n=== Optimizer Parameter Groups ===")
        print(alpha_params)

        if alpha_params:
            optimizer = torch.optim.AdamW(
                [
                    {"params": other_params, "lr": lr, "weight_decay": weight_decay},
                    {"params": alpha_params, "lr": lr * alpha_lr_mult, "weight_decay": 0.0},
                ]
            )
            print(
                f"Using separate alpha LR: base_lr={lr}, alpha_lr={lr * alpha_lr_mult}, "
                f"alpha_params={len(alpha_params)}"
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            print("No alpha parameters found; using single optimizer group.")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    for epoch in range(1, epochs + 1):
        loss, acc = train_epoch(
            model,
            trainloader,
            optimizer,
            device,
            lambda_cost,
            log,
            model_id,
            cost_reduction=cost_reduction,
        )
        test_loss, test_acc = evaluate(model, testloader, device, log, model_id)
        if log:
            log_dict = {
                "train/loss": loss,
                "train/acc": acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
            }
            if use_quant:
                 for name, module in model.named_modules():
                    if isinstance(module, ModuleQuantizer):
                        idx = module.alpha.argmax().item()
                        current_bit = module.bit_choices[idx]
                        log_dict[f"bits/{name}"] = current_bit
            wandb.log(log_dict)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    if use_quant:
        # iterate through all modules write chosen bit from ModuleQuantizer to each layer
        print("\n=== Final Layer Bitwidths ===")
        for name, module in model.named_modules():
            if isinstance(module, ModuleQuantizer):
                chosen_bit = module.finalize_choice()
                print(f"Layer {name}: {chosen_bit} bits")

    # Save final checkpoint
    print("\n=== Saving Final Checkpoint ===")
    quantization_metadata = extract_quantization_metadata(model) if use_quant else None

    # Create a simple args dict for the checkpoint function
    training_args = {
        'model_type': model_type,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'lambda_cost': lambda_cost,
        'bit_choices': bit_choices,
        'use_quant': use_quant,
        'cost_reduction': cost_reduction,
        'alpha_lr_mult': alpha_lr_mult,
        'weight_decay': weight_decay,
        'seed': seed,
        'deterministic': deterministic,
        'debug': debug,
        'include_cnn': include_cnn,
        'num_workers': num_workers
    }

    checkpoint_path = save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        loss=loss,
        acc=acc,
        test_loss=test_loss,
        test_acc=test_acc,
        args=training_args,
        quantization_metadata=quantization_metadata
    )

    print(f"\n✓ Training complete! Checkpoint saved to: {checkpoint_path}")
    print(f"  Use this checkpoint path for downstream evaluation:")
    print(f"  python downstream_evaluation.py --model_path {checkpoint_path}/model.pth --bit_choices '{bit_choices}'")


if __name__ == "__main__":
    app()   