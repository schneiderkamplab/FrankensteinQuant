import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb


def _set_quant_runtime(model, hard_select=False):
    for module in model.modules():
        if hasattr(module, "w_q") and hasattr(module, "a_q") and hasattr(module, "hard_select"):
            module.hard_select = hard_select

def train_epoch(model, loader, optimizer, device, lambda_cost, log, model_id="None", cost_reduction="sum"):
    model.train()
    total_loss, total_acc = 0, 0

    # Configure all quantized layers once before the batch loop.
    _set_quant_runtime(model, hard_select=False)

    pbar = tqdm(loader, desc="Training")
    for batch in pbar: 
        if model_id is not None and "t5" in model_id:
            batch = {k: v.to(device) for k, v in batch.items()}
            task_loss = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )["loss"]
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x) 
            task_loss = F.cross_entropy(logits, y)
        cost = 0.0
        cost_modules = 0
    
        for module in model.modules():
            if hasattr(module, 'get_cost'):
                cost += module.get_cost()
                cost_modules += 1

        if cost_reduction == "mean":
            cost_term = cost / max(cost_modules, 1)
        elif cost_reduction == "sum":
            cost_term = cost
        else:
            raise ValueError(f"Unknown cost_reduction: {cost_reduction}. Use 'sum' or 'mean'.")

        loss = task_loss + lambda_cost * cost_term
        
        if model_id is not None and "t5" in model_id:
            pbar.set_postfix({"loss": loss.item()})
        else:
            pbar.set_postfix({"loss": loss.item(), "acc": (logits.argmax(1) == y).float().mean().item()})
        
        if log:
            wandb.log({
                "train/task_loss": task_loss.item(),
                "train/total_loss": loss.item(),
                "train/cost_raw": cost.item() if torch.is_tensor(cost) else float(cost),
                "train/cost_term": cost_term.item() if torch.is_tensor(cost_term) else float(cost_term),
                "train/cost_modules": cost_modules,
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

        if model_id is None or "t5" not in model_id:
            batch_size = x.size(0)
            total_loss += task_loss.item() * batch_size
            total_acc += (logits.argmax(1) == y).sum().item()
    
    pbar.close()
    if model_id is not None and "t5" in model_id:
        return 0.0, 0.0
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

def evaluate(model, loader, device, log, model_id="None"):
    model.eval()
    # Always disable Gumbel sampling for deterministic, apples-to-apples validation.
    _set_quant_runtime(model, hard_select=True)
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Evaluating")

        for batch in pbar:
            if model_id is not None and "t5" in model_id:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )["logits"]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch["target_ids"].view(-1))
                pbar.set_postfix({"loss": loss.item()})
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                pbar.set_postfix({"loss": loss.item(), "acc": (logits.argmax(1) == y).float().mean().item()})
                loss = loss.item() * x.size(0)
                acc = (logits.argmax(1) == y).sum().item()
                total_loss += loss
                total_acc += acc
            
            if log and model_id is not None and "t5" not in model_id:
                wandb.log({
                    "eval/loss": loss,
                    "eval/acc": acc
                })
            elif log and model_id is not None and "t5" in model_id:
                wandb.log({
                    "eval/loss": loss
                })
        pbar.close()
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)