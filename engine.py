import torch.functional as F
from tqdm import tqdm
import wandb

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
