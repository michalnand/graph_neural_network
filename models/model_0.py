import torch
import torch.nn

from .GConv import *

class Create(torch.nn.Module):
    def __init__(self, inputs_count, outputs_count):
        super(Create, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.conv0 = GConv(inputs_count, 32)
        self.act0  = torch.nn.ReLU()

        self.conv1 = GConv(32, 32)
        self.act1  = torch.nn.ReLU()
        
        self.conv2 = GConv(32, outputs_count)

        
        self.conv0.to(self.device)
        self.act0.to(self.device)

        self.conv1.to(self.device)
        self.act1.to(self.device)

        self.conv2.to(self.device)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv0(x, edge_index)
        x = self.act0(x)
        x = torch.nn.functional.dropout(x, p=0.02)

        x = self.conv1(x, edge_index)
        x = self.act1(x)
        x = torch.nn.functional.dropout(x, p=0.02)

        x = self.conv2(x, edge_index)
       
        return x

