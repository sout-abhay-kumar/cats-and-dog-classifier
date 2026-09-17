import torch.nn as nn


class DogCatCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 128 × 128 × 3
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 64 × 64 × 32
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 32 × 32 × 64
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            # 16 × 16 × 128
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256 * 8 * 8,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                2
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x