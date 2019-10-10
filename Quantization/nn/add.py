import torch

class Add2(torch.nn.Module):
    def __init__(self):
        super(Add2, self).__init__()
    def forward(self, input):
        return input[0] + input[1]
