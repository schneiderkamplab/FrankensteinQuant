import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


class ViTWrapper(nn.Module):
    def __init__(self, num_classes=100, image_size=32, patch_size=4, hidden_size=192, num_layers=6, num_heads=3):
        super().__init__()
        config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=3,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_labels=num_classes,
        )
        self.vit = ViTForImageClassification(config)
    
    def forward(self, x, tau=1.0, collect_costs=False):
        outputs = self.vit(x)
        return outputs.logits 