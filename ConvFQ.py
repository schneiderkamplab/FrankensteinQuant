
import torch
import torch.nn as nn
import torch.nn.functional as F
from CostMixin import CostMixin
from gumbel_bit_quantizer import GumbelBitQuantizer

class ConvFQ(nn.Module, CostMixin):
    def __init__(
        self,
        in_ch, 
        out_ch, 
        kernel_size, 
        stride=1, 
        padding=0, 
        name="conv",
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.w_q = GumbelBitQuantizer(name=f"{name}_w")
        self.a_q = GumbelBitQuantizer(name=f"{name}_a")

    def forward(self, x, tau=1.0, collect_costs=False):
        xq, c1, _ = self.a_q(x, tau, return_cost=collect_costs) # activation 
        wq, c2, _ = self.w_q(self.conv.weight, tau, return_cost=collect_costs) # weights
        out = F.conv2d(xq, wq, self.conv.bias, stride=self.conv.stride,
                    padding=self.conv.padding)
        # rescale?        
        self.costs = {"act": c1["expected_cost"], "w": c2["expected_cost"]}

        return out

    def __repr__(self):
        return f"ConvFQ(in_ch={self.conv.in_channels}, out_ch={self.conv.out_channels}, kernel_size={self.conv.kernel_size}, stride={self.conv.stride}, padding={self.conv.padding})"

    def get_cost(self):
        total_cost = sum(self.costs.values()) 
        return total_cost