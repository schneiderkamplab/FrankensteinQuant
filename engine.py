import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from LinearFQ import LinearFQ

def train_epoch(model, loader, optimizer, device, tau, lambda_cost, log):
    model.train()
    total_loss, total_acc = 0, 0

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        logits = model(x) 
        task_loss = F.cross_entropy(logits, y)
        print("task_loss:", task_loss.item())
        cost = 0.0
    
        for module in model.modules():
            if hasattr(module, 'get_cost'):
                cost += module.get_cost()
        loss = task_loss + lambda_cost * cost
        
        if log:
            wandb.log({
                "train/task_loss": task_loss.item(),
                "train/total_loss": loss.item(),
                "train/cost": cost, "train/tau": tau
                })
        loss.backward()
        # with torch.no_grad():
        #     for name, module in model.named_modules():
        #         # if isinstance(module, LinearFQ):
        #         if hasattr(module, 'weight'):
        #             if module.weight.grad is not None:
        #                 print(f"{name} grad norm: {module.weight.grad.norm():.6f}")
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * x.size(0)
        total_acc += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)
