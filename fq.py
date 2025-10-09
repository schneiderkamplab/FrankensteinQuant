import torch
import wandb
import typer
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------
# Candidate bitwidths & cost table
# -----------------------
BIT_CHOICES = [2, 4, 8, 16]
COST_TABLE = {2: 0.5, 4: 1.0, 8: 2.0, 16: 3.0}  # example proxy cost

app = typer.Typer()

def gumbel_softmax(logits, tau=1.0, hard=True, eps=1e-20):
    g = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
    y = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        y_hard = F.one_hot(y.argmax(dim=-1), num_classes=logits.size(-1)).float()
        y = (y_hard - y).detach() + y
    return y

class GumbelBitQuantizer(nn.Module):
    def __init__(self, bit_choices=BIT_CHOICES, name=""):
        super().__init__()
        self.bit_choices = bit_choices
        self.K = len(bit_choices)
        self.alpha = nn.Parameter(torch.zeros(self.K), requires_grad=True)  # learnable logits
        self.name = name
        self.chosen_bit = None  # will be filled after training


    def _quantize(self, x, bit):
        qmin, qmax = -(2 ** (bit - 1)), (2 ** (bit - 1)) - 1
        scale = x.detach().abs().max() / qmax
        scale = scale.clamp(min=1e-8)
        qx = torch.clamp((x / scale).round(), qmin, qmax)
        return qx * scale

    def forward(self, x, tau=1.0, return_cost=False):
        probs = gumbel_softmax(self.alpha, tau=tau, hard=True)
        idx = probs.argmax().item()  # hard choice
        bit = self.bit_choices[idx]
        wandb.log({f"{self.name}_bit_choice": bit})
        xq = self._quantize(x, bit)

        costs = {}
        if return_cost:
            expected_cost = sum(
                probs[k] * COST_TABLE[self.bit_choices[k]]
                for k in range(self.K)
            )
            costs = {"expected_cost": expected_cost}
        return xq, costs, probs

    def finalize_choice(self):
        idx = self.alpha.argmax().item()
        self.chosen_bit = self.bit_choices[idx]
        return self.chosen_bit

class QConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, name="conv"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.w_q = GumbelBitQuantizer(name=f"{name}_w")
        self.a_q = GumbelBitQuantizer(name=f"{name}_a")

    def forward(self, x, tau=1.0, collect_costs=False):
        xq, c1, _ = self.a_q(x, tau, return_cost=collect_costs)
        wq, c2, _ = self.w_q(self.conv.weight, tau, return_cost=collect_costs)
        out = F.conv2d(xq, wq, self.conv.bias, stride=self.conv.stride,
                       padding=self.conv.padding)
        costs = {}
        if collect_costs:
            costs = {"act": c1["expected_cost"], "w": c2["expected_cost"]}
        return out, costs

class QLinear(nn.Module):
    def __init__(self, in_f, out_f, name="fc"):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.w_q = GumbelBitQuantizer(name=f"{name}_w")
        self.a_q = GumbelBitQuantizer(name=f"{name}_a")

    def forward(self, x, tau=1.0, collect_costs=False):
        xq, c1, _ = self.a_q(x, tau, return_cost=collect_costs)
        wq, c2, _ = self.w_q(self.linear.weight, tau, return_cost=collect_costs)
        out = F.linear(xq, wq, self.linear.bias)
        costs = {}
        if collect_costs:
            costs = {"act": c1["expected_cost"], "w": c2["expected_cost"]}
        return out, costs

class SmallQNet(nn.Module):
    def __init__(self, lr=0.001, wd=0.0, num_classes=100, use_quant=True):
        super().__init__()
        self.use_qaunt = use_quant
        self.lr = lr
        self.wd = wd
        self.flatten = nn.Flatten()

        if use_quant:
            self.conv1 = QConv2d(3, 32, kernel_size=3, stride=1, padding=1, name="conv1")
            self.conv2 = QConv2d(32, 64, kernel_size=3, stride=1, padding=1, name="conv2")
            self.conv3 = QConv2d(64, 128, kernel_size=3, stride=1, padding=1, name="conv2")
            self.fc1 = QLinear(128 * 4 * 4, 512, name="fc")
            self.fc2 = QLinear(512, num_classes, name="fc")
        else:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(128 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, num_classes)
        

    def forward(self, x, tau=1.0, collect_costs=False):
        total_cost = 0.0

        x, cost = self.conv1(x, tau, collect_costs)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        total_cost += sum(cost.values()) if collect_costs else 0

        x, cost = self.conv2(x, tau, collect_costs)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        total_cost += sum(cost.values()) if collect_costs else 0

        x, cost = self.conv3(x, tau, collect_costs)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        total_cost += sum(cost.values()) if collect_costs else 0

        # MLP
        x = self.flatten(x)
        x, cost = self.fc1(x, tau, collect_costs)
        x = F.relu(x)
        logits = self.fc2(x, tau, collect_costs)

        return logits, total_cost 

    def finalize_bitwidths(self):
        bit_config = {}
        for name, module in self.named_modules():
            if isinstance(module, GumbelBitQuantizer):
                bit = module.finalize_choice()
                bit_config[module.name] = bit
        return bit_config


def train_epoch(model, loader, optimizer, device, tau, lambda_cost):
    model.train()
    total_loss, total_acc = 0, 0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        logits, cost = model(x) #, tau, collect_costs=True)
        logits = logits[0] if isinstance(logits, tuple) else logits
        task_loss = F.cross_entropy(logits, y)
        loss = task_loss + lambda_cost * cost
        wandb.log({
            "train/task_loss": task_loss.item(),
            "train/total_loss": loss.item(),
            "train/cost": cost, "train/tau": tau
            })
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


@app.command()
def main(
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    lambda_cost: float = 0.001,
    tau_start: float = 5.0,
    tau_end: float = 0.5,
    use_quant: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    wandb.init(project="frankenstein-quant", name="smallqnet-cifar100")
    wandb.config.update({
        "epochs": 10,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "bit_choices": BIT_CHOICES,
        "cost_table": COST_TABLE,
        "use_quant": use_quant,
    })
    model = SmallQNet(use_quant=use_quant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        tau = tau_start * (tau_end / tau_start) ** (epoch / (epochs - 1))
        loss, acc = train_epoch(model, trainloader, optimizer, device, tau, lambda_cost)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}, tau={tau:.2f}")


    bit_config = model.finalize_bitwidths()
    print("\n=== Final Layer Bitwidths ===")
    for k, v in bit_config.items():
        print(f"{k}: {v} bits")


if __name__ == "__main__":
    app()
