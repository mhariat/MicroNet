import torch
import math
from nn.quantization.quantmodule import QuantModule


class SigmaMax(QuantModule):
    r"""Quantizes the tensor using the ADD SCHEME DESCRIPTION quantization

    The quantization is defined by the mapping between the real value,
    :math:`f`, and the quantized value, :math:`q`:

    .. math::
        \begin{equation*}
        f = 2^{-B}\times q,
        \end{equation*}

    where :math:`B` is the binary point.

    Given a float tensor, :math:`T_{\text{f}}`, the quantized tensor,
    :math:`T_{\text{q}}`, using :math:`W` bits is computed as:

    .. math::
        \begin{equation*}
        f_{\text{max}} = \left|\mu_{T_{\text{f}}}\right| + \lambda\sigma_{T_{\text{f}}}
        \end{equation*}

    .. math::
        \begin{equation*}
        M = \left\lfloor \log_{2}f_{\text{max}} \right\rfloor + 1
        \end{equation*}

    .. math::
        \begin{equation*}
        B = W - 1 - M,
        \end{equation*}

    .. math::
        \begin{equation*}
        T_{\text{q}} = \text{round}\left(\frac{T_{\text{f}}}{2^{-B}\right),
        \end{equation*}

    Args:
        forward_bits (int): bit-width of the forward tensor.
        forward_avg_const (float): weight for calculating running
            exponential averages of forward pass tensor mean and std values.
        forward_num_std (int): factor of std used to compute maximum for
            forward tensor while profiling.
        backward_bits (int): bit-widht of the backward (gradient) tensor.
            Default: ``None``
        backward_avg_const (float): weight for calculating running
            exponential averages of gradient tensor mean and std values.
            Default: ``None``
        backward_num_std (int): factor of std used to compute maximum for
            backward tensor while profiling.
            Default: ``None``

    """

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
        res = 1
        n = len(input.shape)
        for k in range(n):
            res *= input.shape[k]

        if res == 1:
            std = 1e-15 * torch.abs(mean)
        else:
            std = input.detach().std()

        # if len(input.shape) == 4:
        #     if input.shape[0]*input.shape[1]*input.shape[2]*input.shape[3] == 1:
        #         std = 1e-15*torch.abs(mean)
        #     else:
        #         std = input.detach().std()
        # else:
        #     std = input.detach().std()

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
        '''
        '''
        ctx.module = module
        if not module.forward_bits:
            return input

        # Update forward quantization parameters if required
        module.update_forward_params(input)

        bp, qmin, qmax = module.forward_params()
        return SigmaMaxQuantFunction.quantize(input, bp, qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        '''
        module = ctx.module
        if not module.backward_bits:
            return grad_output, None

        # Update backward quantization parameters if required
        module.update_backward_params(grad_output)

        bp, qmin, qmax = module.backward_params()
        grad_input = SigmaMaxQuantFunction.quantize(grad_output, bp, qmin, qmax)

        return grad_input, None

    @staticmethod
    def quantize(input, bp, qmin, qmax):
        delta = 2. ** (-bp)
        # Quantize
        output = input.div(delta)
        output.clamp_(qmin, qmax).round_()
        # Dequantize
        output.mul_(delta)
        return output

    @staticmethod
    def symbolic(g, input, module):
        '''
        '''
        bp, qmin, qmax = module.forward_params()
        kwargs = {"versipoint_i": [module.forward_bits, bp]}
        quant_node = g.op("Quant", input, **kwargs)

        return quant_node

