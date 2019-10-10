import math
import copy
from torch.nn.parameter import Parameter
from collections import OrderedDict
from utils.common_utils import *


class QuantModule(torch.nn.Module):
    def __init__(self, forward_bits=None, backward_bits=None):
        super(QuantModule, self).__init__()
        self.forward_bits = forward_bits
        self.backward_bits = backward_bits

        self.profiling = False
        self.quant_function = None

    def forward(self, *input):
        raise NotImplementedError

    def profile(self, mode=True):
        self.profiling = mode
        return self


class SigmaMax(QuantModule):
    def __init__(self, forward_bits, forward_avg_const, forward_num_std,
                 backward_bits=None, backward_avg_const=None, backward_num_std=None):
        super(SigmaMax, self).__init__(forward_bits, backward_bits)

        self.forward_avg_const = forward_avg_const
        self.backward_avg_const = backward_avg_const

        self.forward_qmin = None
        self.forward_qmax = None
        if self.forward_bits:
            bound = 2 ** (self.forward_bits - 1)
            self.forward_qmin = -bound
            self.forward_qmax = bound - 1

        self.backward_qmin = None
        self.backward_qmax = None
        if self.backward_bits:
            bound = 2 ** (self.backward_bits - 1)
            self.backward_qmin = -bound
            self.backward_qmax = bound - 1

        self.forward_mean = None
        self.forward_std = None
        self.forward_max = None
        self.forward_min = None
        self.forward_num_std = forward_num_std if self.forward_bits else None

        self.backward_mean = None
        self.backward_std = None
        self.backward_num_std = backward_num_std if self.backward_bits else None

        self.register_buffer('forward_bp', None)
        self.register_buffer('backward_bp', None)

    def forward(self, input):
        return SigmaMaxQuantFunction.apply(input, self)

    def update_forward_params(self, input):
        if not self.profiling:
            if self._buffers['forward_bp'] is None:
                raise ValueError('Forward quantization binary point (bp) is not defined.')
            return

        mean = input.detach().mean()
        if len(input.shape) == 4:
            if input.shape[0]*input.shape[1]*input.shape[2]*input.shape[3] == 1:
                std = 1e-15*torch.abs(mean)
            else:
                std = input.detach().std()
        else:
            std = input.detach().std()

        if self.forward_mean is None or self.forward_std is None:
            self.forward_mean = mean
            self.forward_std = std

        else:
            self.forward_mean.mul_(1. - self.forward_avg_const).add_(mean * self.forward_avg_const)
            self.forward_std.mul_(1. - self.forward_avg_const).add_(std * self.forward_avg_const)

        self.forward_max = input.detach().abs().max()
        self.forward_min = input.detach().abs().min()

        bp = SigmaMax.binary_point(self.forward_mean, self.forward_std, self.forward_num_std, self.forward_bits)

        if self._buffers['forward_bp'] is None:
            self._buffers['forward_bp'] = torch.tensor(1, dtype=torch.int, device=input.device)

        self._buffers['forward_bp'].fill_(bp)

        return

    def update_backward_params(self, input):
        if not self.profiling:
            if self._buffers['backward_bp'] is None:
                raise ValueError('Backward quantization binary point (bp) is not defined.')
            return

        mean = input.detach().mean()
        std = input.detach().std()

        if self.backward_mean is None or self.backward_std is None:
            self.backward_mean = mean
            self.backward_std = std
        else:
            self.backward_mean.mul_(1. - self.backward_avg_const).add_(mean * self.backward_avg_const)
            self.backward_std.mul_(1. - self.backward_avg_const).add_(std * self.backward_avg_const)

        bp = SigmaMax.binary_point(self.backward_mean, self.backward_std, self.backward_num_std, self.backward_bits)

        if self._buffers['backward_bp'] is None:
            self._buffers['backward_bp'] = torch.tensor(1, dtype=torch.int, device=input.device)

        self._buffers['backward_bp'].fill_(bp)

        return

    def forward_params(self):
        return self._buffers['forward_bp'].item(), self.forward_qmin, self.forward_qmax

    def backward_params(self):
        return self._buffers['backward_bp'].item(), self.backward_qmin, self.backward_qmax

    @staticmethod
    def binary_point(mean, std, num_std, total_bits):
        f_max = mean.abs() + num_std * std
        M = math.ceil(math.log2(f_max + 1.0e-12))  # math.floor(math.log2(f_max)) + 1
        bp = total_bits - 1 - M
        return bp


class SigmaMaxQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, module):
        ctx.module = module
        if not module.forward_bits:
            return input

        # Update forward quantization parameters if required
        module.update_forward_params(input)

        bp, qmin, qmax = module.forward_params()
        return SigmaMaxQuantFunction.quantize(input, bp, qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        module = ctx.module
        if not module.backward_bits:
            return grad_output, None
        module.update_backward_params(grad_output)

        bp, qmin, qmax = module.backward_params()
        grad_input = SigmaMaxQuantFunction.quantize(grad_output, bp, qmin, qmax)

        return grad_input, None

    @staticmethod
    def quantize(input, bp, qmin, qmax):
        delta = 2. ** (-bp)
        output = input.div(delta)
        output.clamp_(qmin, qmax).round_()
        output.mul_(delta)
        return output

    @staticmethod
    def symbolic(g, input, module):
        bp, qmin, qmax = module.forward_params()
        kwargs = {"versipoint_i": [module.forward_bits, bp]}
        quant_node = g.op("Quant", input, **kwargs)

        return quant_node


class EmptyLayer(torch.nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


def insert_quant_modules(model, a_quant_module):
    condition = True
    quant_modules = []
    for k, v in model._modules.items():
        if len(v._modules.items()) > 0:
            modules, __ = insert_quant_modules(v, a_quant_module)
            for module in modules:
                quant_modules.append(module)
        else:
            if not isinstance(v, EmptyLayer):
                if condition:
                    new_modules = OrderedDict()
                    new_modules[k] = v
                    new_quant_module = copy.deepcopy(a_quant_module)
                    new_modules['quant'] = new_quant_module
                    quant_modules.append(new_quant_module)
                    model._modules[k] = torch.nn.Sequential(new_modules)

    return quant_modules, model


class QuantizedNet(torch.nn.Module):

    def __init__(self, net, a_quant_module, w_quant_module):
        super(QuantizedNet, self).__init__()

        self.quant_input = copy.deepcopy(a_quant_module)
        self.a_quant_modules = []
        self.a_quant_modules, self.quant_net = insert_quant_modules(copy.deepcopy(net), a_quant_module)
        self.a_quant_modules.append(self.quant_input)

        self.w_quant_modules = {}

        for name, _ in self.quant_net.named_parameters():
            self.w_quant_modules[name] = copy.deepcopy(w_quant_module)

        self.profiling = False

    def forward(self, input):
        return self.quant_net(self.quant_input(input))

    def train(self, mode=True):
        super(QuantizedNet, self).train(mode)
        self._profile(mode)

        return self

    def eval(self):
        return self.train(False)

    def to(self, *args, **kwargs):
        super(QuantizedNet, self).to(*args, **kwargs)

        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        if device is not None:
            for name, param in self.quant_net.named_parameters():
                quant_module = self.w_quant_modules[name]
                quant_module.to(device)

    def _profile(self, mode=True):
        """Sets the module in profiling mode.

        In the profiling mode, at every forward and backward pass
        the quantization parameters and statistics are updated
        by all the activation and weight quantization modules.

        Returns:
            Module: self
        """
        self.profiling = mode
        # Set profile flag for weight quant modules
        for name, param in self.quant_net.named_parameters():
            quant_module = self.w_quant_modules[name]
            quant_module.profile(mode)

        # Set profile flag for activation quant modules
        for quant_module in self.a_quant_modules:
            quant_module.profile(mode)

        return self


def load_quantization_architecture(model):
    for module in model.modules():
        if isinstance(module, SigmaMax):
            module.register_buffer('forward_bp', torch.tensor(0))
