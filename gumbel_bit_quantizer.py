import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelBitQuantizer(nn.Module):
    def __init__(self, bit_choices, cost_table, name="", bias=None, device=None):
        super().__init__()
        self.K = len(bit_choices)
        self.bit_choices = bit_choices
        # Start from an unbiased state so runs do not diverge from random alpha init.
        self.alpha = nn.Parameter(torch.zeros(self.K), requires_grad=True)
        self.name = name
        self.cost_table = cost_table
        self.chosen_bit = None  # will be filled after training
        self.bias = bias # parsed in for now to satisfy argument passing
        self.device = device # parsed in for now to satisfy argument passing

    def _quantize(self, x, bit):
        bit = int(round(bit.item()))  # Convert to int for quantization
        qmin = -(2 ** (bit - 1))
        qmax = (2 ** (bit - 1)) - 1
        # Use mean of abs for better gradient flow instead of max
        #scale = x.abs().mean() * 2.5 / qmax  # 2.5 factor to cover most of the range
        scale = x.abs().mean() / qmax  # 2.5 factor to cover most of the range
        scale = scale.clamp(min=1e-8)
        # Straight-through estimator: forward uses round, backward uses identity
        xq = torch.clamp(x / scale, qmin, qmax)
        xq_rounded = xq.round()
        assert (xq_rounded >= qmin).all() and (xq_rounded <= qmax).all(), f"Quantized values out of range: {xq_rounded.min().item()} to {xq_rounded.max().item()}, expected [{qmin}, {qmax}], with bit: {bit}"
        xq = (xq_rounded - xq).detach() + xq  # STE: forward=round, backward=identity
        return xq * scale, scale

    def forward(self, x, return_cost=False, verbose=False, hard_select=False):
        probs = F.softmax(self.alpha, dim=-1)

        if hard_select:
            # Discrete deployment-style selection: evaluate with a single chosen bit-depth.
            if self.chosen_bit is not None and self.chosen_bit in self.bit_choices:
                chosen_idx = self.bit_choices.index(self.chosen_bit)
            else:
                chosen_idx = int(self.alpha.argmax().item())
            probs = F.one_hot(
                torch.tensor(chosen_idx, device=probs.device),
                num_classes=self.K,
            ).to(dtype=probs.dtype)

        bit_choices_t = torch.tensor(self.bit_choices, device=probs.device, dtype=probs.dtype)
        bit_soft = torch.sum(probs * bit_choices_t)
        if verbose:
            print(f"{self.name} - bit_soft: {bit_soft.item():.2f}, probs: {probs.detach().cpu().numpy()}, alpha: {self.alpha.detach().cpu().numpy()}")
            print(f"soft bit {bit_soft.item():.4f}")

        # Differentiable bit selection: mix quantized tensors from all candidate bit-widths.
        # This preserves task gradients to `probs` / `alpha`, unlike hard-casting bit_soft to int.
        xq_mix = 0.0
        scale_mix = 0.0
        for k, bit in enumerate(self.bit_choices):
            xq_k, scale_k = self._quantize(x, x.new_tensor(float(bit)))
            xq_mix = xq_mix + probs[k] * xq_k
            scale_mix = scale_mix + probs[k] * scale_k

        xq, scale = xq_mix, scale_mix
        if verbose:
            print("Quantized output stats - mean: {:.4f}, std: {:.4f}".format(xq.mean().item(), xq.std().item()))

        costs = {}
        if return_cost:
            expected_cost = sum(
                probs[k] * self.cost_table[self.bit_choices[k]]
                for k in range(self.K)
            )
            costs = {"expected_cost": expected_cost}

        return xq, costs, probs, scale

    def finalize_choice(self):
        idx = self.alpha.argmax().item()
        self.chosen_bit = self.bit_choices[idx]
        return self.chosen_bit
    
"""
nvidia-ml-py
pip install torch-c-dlpack-ext
"""