import torch
import torch.nn as nn
import torch.nn.functional as F

from gumbel_bit_quantizer import GumbelBitQuantizer

class LinearFQ(nn.Module):
    def __init__(
        self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            name="fc",
    ):
        super(LinearFQ, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.bit_choices = [2, 4, 8]
        self.K = len(self.bit_choices)
        self.COST_TABLE = {2: 0.5, 4: 1.0, 8: 2.0, 16: 3.0}  # example proxy cost        
        self.w_q = GumbelBitQuantizer(name=f"{name}_w")
        self.a_q = GumbelBitQuantizer(name=f"{name}_a")

    def __repr__(self):
        return f"LinearFQ(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

    def forward(self, x, tau=1.0, collect_costs=False):
        xq, c1, _ = self.a_q(x, tau, return_cost=collect_costs)
        wq, c2, _ = self.w_q(self.linear.weight, tau, return_cost=collect_costs)
        out = F.linear(xq, wq, self.linear.bias)
        costs = {}
    
        if collect_costs:
            costs = {"act": c1["expected_cost"], "w": c2["expected_cost"]}
    
        return out, costs
    
