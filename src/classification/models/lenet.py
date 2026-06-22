import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 10
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# The formula for the output size (spatial dimensions) of a convolutional layer is:
#   output_size = (input_size - kernel_size + 2*padding) / stride + 1
# Input 3x28x28
#   conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2, stride=1)
#       (28 - 5 + 2 * 2) / 1 + 1 = 28 -> 6x28x28
#   avg_pool2d(x, kernel_size=2, stride=2)
#       (28 - 2) / 2 + 1 = 14 -> 6x14x14
#   conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
#       (14 - 5) / 1 + 1 = 10 -> 16x10x10
#   avg_pool2d(x, kernel_size=2, stride=2)
#       (10 - 2) / 2 + 1 = 5 -> 16x5x5
#   view(x.size(0), -1) - Flatten
#       16 * 5 * 5 = 400