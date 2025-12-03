import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from LinearFQ import LinearFQ

def train_epoch(model, loader, optimizer, device, tau, lambda_cost, log):
    model.train()
    total_loss, total_acc = 0, 0

    pbar = tqdm(loader, desc="Training")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        logits = model(x) 
        task_loss = F.cross_entropy(logits, y)
        cost = 0.0
    
        for module in model.modules():
            if hasattr(module, 'get_cost'):
                cost += module.get_cost()
        loss = task_loss + lambda_cost * cost
        
        pbar.set_postfix({"loss": loss.item(), "entropy": task_loss.item(), "acc": (logits.argmax(1) == y).float().mean().item()})
        if log:
            wandb.log({
                "train/task_loss": task_loss.item(),
                "train/total_loss": loss.item(),
                "train/cost": cost, "train/tau": tau
                })
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.update(1)

        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

def evaluate(model, loader, device, log=False):
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if log:
                wandb.log({
                    "eval/loss": loss.item(),
                })
            total_loss += loss.item() * x.size(0)
            total_acc += (logits.argmax(1) == y).sum().item()
            pbar.set_postfix({"loss": loss.item(), "acc": (logits.argmax(1) == y).float().mean().item()})
            pbar.update(1)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)