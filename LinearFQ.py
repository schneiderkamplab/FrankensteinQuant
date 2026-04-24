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
        self.w_q = GumbelBitQuantizer(name=f"{name}_w", **kwargs)
        self.a_q = GumbelBitQuantizer(name=f"{name}_a", **kwargs)
        # External trainer updates this each step; used when forward is called without tau.
        self.tau = 1.0
        self.use_gumbel = True
        self.hard_select = False

    def __repr__(self):
        return f"LinearFQ(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"

    def forward(self, x, tau=None, collect_costs=True, rescale=True):
        if tau is None:
            tau = self.tau
        x_quant, c1, _, scale1 = self.a_q(
            x,
            tau,
            return_cost=collect_costs,
            use_gumbel=self.use_gumbel,
            hard_select=self.hard_select,
        )
        w_quant, c2, _, scale2 = self.w_q(
            self.weight,
            tau,
            return_cost=collect_costs,
            use_gumbel=self.use_gumbel,
            hard_select=self.hard_select,
        )
        out = F.linear(x_quant, w_quant, self.bias)
        
        # No need to rescale here - quantization already handles it
        # The quantized values are already rescaled back to original range

        self.costs = {"act": c1["expected_cost"], "w": c2["expected_cost"]}
    
        return out
    
    def get_cost(self):
        total_cost = sum(self.costs.values()) 
        return total_cost