import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 2
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Layer 3
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# The formula for the output size (spatial dimensions) of a convolutional layer is:
#   output_size = (input_size - kernel_size + 2*padding) / stride + 1
# Input 3x227x227
#   nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
#       (227 - 11 + 2 * 0) / 4 + 1 = 55 -> 96x55x55
#   nn.MaxPool2d(kernel_size=3, stride=2)
#       (55 - 3) / 2 + 1 = 27 -> 64x27x27
#   nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
#       (27 - 5 + 2 * 2) / 1 + 1 = 27 -> 256x27x27
#   nn.MaxPool2d(kernel_size=3, stride=2)
#       (27 - 3) / 2 + 1 = 13 -> 256x13x13
#   nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
#       (13 - 3 + 2 * 1) / 1 + 1 = 13 -> 384x13x13
#   nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
#       (13 - 3 + 2 * 1) / 1 + 1 = 13 -> 384x13x13
#   nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
#       (13 - 3 + 2 * 1) / 1 + 1 = 13 -> 256x13x13
#   nn.MaxPool2d(kernel_size=3, stride=2)
#       (13 - 3) / 2 + 1 = 6 -> 256x6x6
