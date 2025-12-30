import torch.nn as nn
import torch.nn.functional as F

class SmallNet(nn.Module):
    def __init__(self, lr=0.001, wd=0.0, num_classes=100):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.flatten = nn.Flatten()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x, tau=1.0, collect_costs=False):
        total_cost = 0.0

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # MLP
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)

        return logits