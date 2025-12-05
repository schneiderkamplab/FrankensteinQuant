import ssl
import typer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import wandb

from ConvFQ import ConvFQ
from engine import train_epoch, evaluate
from fq import frankensteinize
from gumbel_bit_quantizer import GumbelBitQuantizer
from transformers import ViTForImageClassification, ViTConfig

ssl._create_default_https_context = ssl._create_unverified_context
app = typer.Typer()

BIT_CHOICES = [2, 4, 8, 16]
COST_TABLE = {2: 0.5, 4: 1.0, 8: 2.0, 16: 3.0}  # example proxy cost

class SmallNet(nn.Module):
    def __init__(self, lr=0.001, wd=0.0, num_classes=100):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, tau=1.0, collect_costs=False):
        total_cost = 0.0

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # MLP
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)

        return logits


class ViTWrapper(nn.Module):
    def __init__(self, num_classes=100, image_size=32, patch_size=4, hidden_size=192, num_layers=6, num_heads=3):
        super().__init__()
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_labels=num_classes,
        )
        self.vit = ViTForImageClassification(config)
    
    def forward(self, x, tau=1.0, collect_costs=False):
        outputs = self.vit(x)
        return outputs.logits 


@app.command()
def main(
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    lambda_cost: float = 0.001,
    tau_start: float = 5.0,
    tau_end: float = 0.5,
    use_quant: bool = False,
    bit_choices: str = None,
    log: bool = False,
    model_type: str = "vit",  # "smallnet" or "vit"
):
    if use_quant:
        if bit_choices is not None:     
            bit_choices = eval(bit_choices)
        else:
            # typer.echo("Using default bit choices:", BIT_CHOICES)
            bit_choices = BIT_CHOICES

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    if log:
        run_name = f"{model_type.upper()}_CIFAR100" + (f"_Quant_{str(bit_choices)}" if use_quant else "_FullPrec")
        wandb.init(project="frankenstein-quant", name=run_name)
    
    # Select model
    if model_type.lower() == "vit":
        model = ViTWrapper(num_classes=100).to(device)
    else:
        model = SmallNet().to(device)

    if use_quant:
        typer.echo("Applying Frankenstein Quantization...")
        model = frankensteinize(model, new_class_kwargs={
            "name": "fc",
            "bit_choices": bit_choices,
            "cost_table": COST_TABLE
        })
        # model = frankensteinize(model, old_class=nn.Conv2d, new_class=ConvFQ, new_class_kwargs={
        #     "name": "conv2d",
        #     "bit_choices": bit_choices,
        #     "cost_table": COST_TABLE
        # })

    print("Model Summary:")
    print(model)
    model.to(device)

    if log:
         wandb.config.update({
            "epochs": 10,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "bit_choices": bit_choices,
            "cost_table": COST_TABLE,
            "use_quant": use_quant,
            "model": str(model)
        })
    # Note always to layer replacements BEFORE optimizer creation 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        tau = tau_start * (tau_end / tau_start) ** (epoch / (epochs - 1))
        loss, acc = train_epoch(model, trainloader, optimizer, device, tau, lambda_cost, log)
        test_loss, test_acc = evaluate(model, testloader, device, log)
        if log:
            wandb.log({
                "train/loss": loss,
                "train/acc": acc,
                "test/loss": test_loss,
                "test/acc": test_acc,
                "train/tau": tau
            })
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, tau={tau:.2f}")

    # Finalize bitwidth choices
    if use_quant:
        # iterate through all modules write chosen bit from GumbelBitQuantizer to each layer
        print("\n=== Final Layer Bitwidths ===")
        for name, module in model.named_modules():
            if isinstance(module, GumbelBitQuantizer):
                chosen_bit = module.finalize_choice()
                print(f"Layer {name}: {chosen_bit} bits")


if __name__ == "__main__":
    app()   