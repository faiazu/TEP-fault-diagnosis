import torch.nn as nn

# simple feed forward neural network
# for learning loop
# layer 1, 3120 numer vector -> 256 num vector
# ReLu activation function, faster than sigmoid
# layer 2, 256 num vector -> 128 num vector
# layer 3, 128 num vector -> # answers long vector
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
