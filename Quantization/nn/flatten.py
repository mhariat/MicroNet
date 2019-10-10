import torch

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        if(input.dim() > 2):
            return FlattenFunction.apply(input)
        else:
            raise TypeError("Input Error: Only 3D or higher dimension input Tensors are supported (got {}D)".format(input.dim()))

class FlattenFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.in_shape = input.shape
        return input.view(input.size(0), -1)
    
    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.in_shape
        return grad_output.view(*shape)

    @staticmethod
    def symbolic(g, input):
        kwargs = {}
        node = g.op("Flatten", input, **kwargs)
        return node
