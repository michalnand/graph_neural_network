import torch
import torch.nn

from .GConv import *


class Create(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count, hidden_count = 32):
        super(Create, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv0      = GConv(inputs_count, hidden_count)        
        self.conv1      = GConv(hidden_count, hidden_count)
        
        self.linear     = torch.nn.Linear(hidden_count, outputs_count)

        self.conv0.to(self.device) 
        self.conv1.to(self.device)
        self.linear.to(self.device)

    def forward(self, position, velocity, force, edge_index):
        x = torch.cat([position, velocity, force], dim=1)

        x = self.conv0(x, edge_index)
        x = torch.nn.functional.relu(x)

        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)

        x = self.linear(x)

        return x

