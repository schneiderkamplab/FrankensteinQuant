
import torch.nn as nn
import torch.nn.functional as F
from CostMixin import CostMixin
from gumbel_bit_quantizer import GumbelBitQuantizer

class ConvFQ(nn.Conv2d, CostMixin):
    def __init__(
        self,
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        name="conv",
        **kwargs
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.w_q = GumbelBitQuantizer(name=f"{name}_w", **kwargs)
        self.a_q = GumbelBitQuantizer(name=f"{name}_a", **kwargs)
        self.hard_select = False

    def forward(self, x, collect_costs=True):
        x_quant, c1, _, scale1 = self.a_q(
            x,
            return_cost=collect_costs,
            hard_select=self.hard_select,
        ) # activation
        w_quant, c2, _, scale2 = self.w_q(
            self.weight,
            return_cost=collect_costs,
            hard_select=self.hard_select,
        ) # weights
        out = F.conv2d(x_quant, w_quant, self.bias, stride=self.stride, padding=self.padding)
        self.costs = {"act": c1["expected_cost"], "w": c2["expected_cost"]}

        return out

    def __repr__(self):
        return f"ConvFQ(in_ch={self.in_channels}, out_ch={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"

    def get_cost(self):
        total_cost = sum(self.costs.values()) 
        return total_cost