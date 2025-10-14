import torch
import wandb
import typer
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from engine import train_epoch
from fq import frankensteinize

import ssl

from gumbel_bit_quantizer import GumbelBitQuantizer
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------
# Candidate bitwidths & cost table
# -----------------------
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

def main(
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    lambda_cost: float = 0.001,
    tau_start: float = 5.0,
    tau_end: float = 0.5,
    use_quant: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    wandb.init(project="frankenstein-quant", name="smallnet-cifar100")
    wandb.config.update({
        "epochs": 10,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "bit_choices": BIT_CHOICES,
        "cost_table": COST_TABLE,
        "use_quant": use_quant,
    })
    model = SmallNet().to(device)

    if use_quant:
        model = frankensteinize(model)
    print("Model Summary:")
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        tau = tau_start * (tau_end / tau_start) ** (epoch / (epochs - 1))
        loss, acc = train_epoch(model, trainloader, optimizer, device, tau, lambda_cost)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}, tau={tau:.2f}")

    # Finalize bitwidth choices
    if use_quant:
        # iterate through all modules write chosen bit from GumbelBitQuantizer to each layer
        print("\n=== Final Layer Bitwidths ===")
        for name, module in model.named_modules():
            if isinstance(module, GumbelBitQuantizer):
                chosen_bit = module.finalize_choice()
                print(f"Layer {name}: {chosen_bit} bits")


if __name__ == "__main__":
    main()
