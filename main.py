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
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import wandb

from ConvFQ import ConvFQ
from engine import train_epoch, evaluate

from models.SmallNet import SmallNet
from models.vit import ViTWrapper
from fq import frankensteinize
from gumbel_bit_quantizer import GumbelBitQuantizer
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
        run_name = f"{model_type.upper()}_CIFAR100" + (f"_Quant_{str(bit_choices)}" if use_quant else "_FullPrec")
        wandb.init(project="scaling-frankenstein-quant-t5", name=run_name+f"_l_{str(lambda_cost)}", tags=f"vit_lambda_{lambda_cost}" if model_type.lower() == "vit" else f"t5_lambda_{lambda_cost}" if model_type.lower() == "t5" else "smallnet")
    
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
            if name.endswith(".alpha"): # learnable logits in GumbelBitQuantizer should have separate LR
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
                    if isinstance(module, GumbelBitQuantizer):
                        idx = module.alpha.argmax().item()
                        current_bit = module.bit_choices[idx]
                        log_dict[f"bits/{name}"] = current_bit
            wandb.log(log_dict)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    if use_quant:
        # iterate through all modules write chosen bit from GumbelBitQuantizer to each layer
        print("\n=== Final Layer Bitwidths ===")
        for name, module in model.named_modules():
            if isinstance(module, GumbelBitQuantizer):
                chosen_bit = module.finalize_choice()
                print(f"Layer {name}: {chosen_bit} bits")


if __name__ == "__main__":
    app()   