import torch
import torch.nn as nn

# simple 1D convolutional neural network
# input should be:
# (batch_size, num_sensors, window_size)
# for this project that means:
# (batch_size, 52, 60)
#
# conv1d slides across time
# while using all sensor channels together
class SimpleCNN1D(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()

        # block 1
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        # block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        # block 3
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # turn the full time dimension into one value per channel
        self.pool = nn.AdaptiveAvgPool1d(1)

        # classifier head
        self.fc1 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.pool(x)

        # pool output shape is (batch_size, 128, 1)
        # flatten it to (batch_size, 128)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x
