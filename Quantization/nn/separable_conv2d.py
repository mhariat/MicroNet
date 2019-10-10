import torch

class SeparableConv2d(torch.nn.Module):
    """Implements depth-seprable Conv2d operation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size,
                                              stride=stride, padding=padding, dilation=dilation,
                                              bias=bias,
                                              groups=in_channels)
        self.pointwise_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                               stride=1, padding=0, dilation=1, bias=bias, 
                                               groups=1)
        
    def forward(self, input):
        out = self.depthwise_conv(input)
        out = self.pointwise_conv(out)
        return out
        



