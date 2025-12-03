import torch
import torch.nn as nn
import torch.nn.functional as F
from CostMixin import CostMixin
from gumbel_bit_quantizer import GumbelBitQuantizer

class LinearFQ(nn.Linear, CostMixin):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        name="fc",
        **kwargs
    ):
        super().__init__(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        # self.COST_TABLE = {2: 0.5, 4: 1.0, 8: 2.0, 16: 3.0}  # example proxy cost        
        # self.bit_choices = [16] #[2, 4, 8, 16]
        self.w_q = GumbelBitQuantizer(name=f"{name}_w", **kwargs)
        self.a_q = GumbelBitQuantizer(name=f"{name}_a", **kwargs)

    def __repr__(self):
        return f"LinearFQ(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

    def forward(self, x, tau=1.0, collect_costs=True, rescale=True):
        x_quant, c1, _, scale1 = self.a_q(x, tau, return_cost=collect_costs)
        w_quant, c2, _, scale2 = self.w_q(self.weight, tau, return_cost=collect_costs)
        out = F.linear(x_quant, w_quant, self.bias)
        
        # No need to rescale here - quantization already handles it
        # The quantized values are already rescaled back to original range

        self.costs = {"act": c1["expected_cost"], "w": c2["expected_cost"]}
    
        return out
    
    def get_cost(self):
        total_cost = sum(self.costs.values()) 
        return total_cost