import torch
import torch.nn as nn

class FusedConvRelu(nn.Module):
    r""" Fused Convolution and ReLU operator"""
    def __init__(self, conv):
        super(FusedConvRelu, self).__init__()

        with torch.no_grad():
            conv_class = type(conv)
            self.conv = conv_class(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=True if conv.bias is not None else False,
                groups=conv.groups
            )
            self.conv.weight.copy_(conv.weight)
            if conv.bias is not None:
                self.conv.bias.copy_(conv.bias)

    def forward(self, input):
        x = self.conv(input)
        x = nn.functional.relu(x)
        return x

class FusedConv2dRelu(nn.Conv2d):
    r""" Fused Convolution and ReLU operator"""
    def __init__(self, conv):
        assert(isinstance(conv, nn.Conv2d))

        in_channels = conv.in_channels
        out_channels = conv.out_channels
        kernel_size = conv.kernel_size
        stride = conv.stride
        padding = conv.padding
        dilation = conv.dilation
        bias = conv.bias
        groups = conv.groups

        super(FusedConv2dRelu, self).__init__(in_channels, out_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=groups,
                                              bias=True if bias is not None else False)
        with torch.no_grad():                                        
            self.weight.copy_(conv.weight)
            if bias is not None:
                self.bias.copy_(conv.bias)

    def forward(self, input):
        x = super(FusedConv2dRelu, self).forward(input)
        x = nn.functional.relu(x)
        return x