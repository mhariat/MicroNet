import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class Upsample(torch.nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # In ONNX to WavegraphIR conversion, scale_factor is 
        # treated as input variable. However, PyTorch does not treat scale_factor
        # as input variable and hence, cannot be accessed via state_dict() routine. 
        # To load weights into wavegraphIR, we need to write scale_factor as weight tensor
        # in the weights state dict. Therefore, to explicitly write scale_factor in the 
        # weight state dict, we are defining dummy parameter called scales which are initialized
        # from the scale_factor. 
        self.scales = Parameter(torch.Tensor(4), requires_grad=False)
        self.scales[0] = 1.0
        self.scales[1] = 1.0
        self.scales[2] = scale_factor
        self.scales[3] = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)