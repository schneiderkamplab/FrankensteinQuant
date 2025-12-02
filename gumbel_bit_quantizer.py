import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
import wandb


class GumbelBitQuantizer(nn.Module):
    def __init__(self, bit_choices, cost_table, name="", bias=None, device=None):
        super().__init__()
        self.K = len(bit_choices)
        self.bit_choices = bit_choices # torch.tensor(bit_choices)
        self.alpha = nn.Parameter(torch.zeros(self.K), requires_grad=True)  # learnable logits
        self.name = name
        # print("cost_table:", cost_table)
        # data = cost_table
        # self.cost_table = TensorDict({}, batch_size=[])
        # self.cost_table["x"] = torch.tensor(list(data.keys()))
        # self.cost_table["y"] = torch.tensor(list(data.values()))
        # print("self.cost_table:", self.cost_table)

        self.cost_table = cost_table
        self.chosen_bit = None  # will be filled after training
        self.bias = bias # parsed in for now to satisfy argument passing
        self.device = device # parsed in for now to satisfy argument passing

    def _quantize(self, x, bit):
        qmin = -(2 ** (bit - 1))
        qmax = (2 ** (bit - 1)) - 1
        # Use mean of abs for better gradient flow instead of max
        scale = x.abs().mean() * 2.5 / qmax  # 2.5 factor to cover most of the range
        scale = scale.clamp(min=1e-8)
        # Straight-through estimator: forward uses round, backward uses identity
        xq = torch.clamp(x / scale, qmin, qmax)
        xq_rounded = xq.round()
        xq = (xq_rounded - xq).detach() + xq  # STE: forward=round, backward=identity
        return xq * scale, scale

    def gumbel_softmax(self, logits, tau=1.0, hard=False, eps=1e-20):
        g = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)
        y = F.softmax((logits + g) / tau, dim=-1)
        if hard:
            y_hard = F.one_hot(y.argmax(dim=-1), num_classes=logits.size(-1)).float()
            y = (y_hard - y).detach() + y
        return y
    
    def forward(self, x, tau=1.0, return_cost=False):
        probs = self.gumbel_softmax(self.alpha, tau=tau, hard=False)
        # print("probs:", probs)
        bit_soft = torch.sum(probs * torch.tensor(self.bit_choices, device=probs.device))
        # print("bit_soft:", bit_soft)

        # idx = probs.argmax()
        # print("idx:", idx)
        # bit = self.bit_choices[idx]
        # self.chosen_bit = bit
        # wandb.log({f"{self.name}_bit_choice": bit})
        xq, scale = self._quantize(x, bit_soft)

        costs = {}
        if return_cost:
            # print("self.cost_table:", self.cost_table)
            expected_cost = sum(
                probs[k] * self.cost_table[self.bit_choices[k]]
                for k in range(self.K)
            )
            costs = {"expected_cost": expected_cost}

        # print(f"xq: {xq}, costs: {costs}, probs: {probs}, scale: {scale}")
        return xq, costs, probs, scale

    def finalize_choice(self):
        idx = self.alpha.argmax().item()
        self.chosen_bit = self.bit_choices[idx]
        return self.chosen_bit
    
