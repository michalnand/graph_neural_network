import torch
import torch.nn

from .GConv import *

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Create(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count, hidden_count = 32):
        super(Create, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv0  = GConv(inputs_count, hidden_count)        
        self.conv1  = GConv(hidden_count, outputs_count)
        
        self.conv0.to(self.device)
        self.conv1.to(self.device)

    def forward(self, x, edge_index):
        x = Flatten(x)

        x = self.conv0(x, edge_index)
        x = torch.nn.functional.relu(x)
      
        x = self.conv1(x, edge_index)
       
        return x

