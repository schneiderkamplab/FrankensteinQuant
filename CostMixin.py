import torch.nn as nn

class CostMixin:
    def __init__(self):
        super().__init__()
        self.cost = 0.0

    def reset_cost(self):
        self.cost = 0.0

    def get_cost(self):
        return self.cost
