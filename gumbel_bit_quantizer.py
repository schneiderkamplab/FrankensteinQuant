import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


class GumbelBitQuantizer(nn.Module):
    def __init__(self, bit_choices, cost_table, name=""):
        super().__init__()
        self.bit_choices = bit_choices
        self.K = len(bit_choices)
        self.alpha = nn.Parameter(torch.zeros(self.K), requires_grad=True)  # learnable logits
        self.name = name
        self.cost_table = cost_table
        self.chosen_bit = None  # will be filled after training

    def _quantize(self, x, bit):
        qmin, qmax = -(2 ** (bit - 1)), (2 ** (bit - 1)) - 1
        scale = x.detach().abs().max() / qmax
        scale = scale.clamp(min=1e-8)
        qx = torch.clamp((x / scale).round(), qmin, qmax)
        return qx * scale

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, hard=True, eps=1e-20):
        g = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        y = F.softmax((logits + g) / tau, dim=-1)
        if hard:
            y_hard = F.one_hot(y.argmax(dim=-1), num_classes=logits.size(-1)).float()
            y = (y_hard - y).detach() + y
        return y
    
    def forward(self, x, tau=1.0, return_cost=False):
        probs = GumbelBitQuantizer.gumbel_softmax(self.alpha, tau=tau, hard=True)
        idx = probs.argmax().item()  # hard choice
        bit = self.bit_choices[idx]
        wandb.log({f"{self.name}_bit_choice": bit})
        xq = self._quantize(x, bit)
    
        costs = {}
        if return_cost:
            expected_cost = sum(
                probs[k] * self.cost_table[self.bit_choices[k]]
                for k in range(self.K)
            )
            costs = {"expected_cost": expected_cost}
        return xq, costs, probs

    def finalize_choice(self):
        idx = self.alpha.argmax().item()
        self.chosen_bit = self.bit_choices[idx]
        return self.chosen_bit
    
