import torch
import math
from nn.quantization.quantmodule import QuantModule


class Max(QuantModule):
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
        f_{\text{max}} = \Max_{T_{\text{f}}}
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
        backward_bits (int): bit-widht of the backward (gradient) tensor.
            Default: ``None``
        backward_avg_const (float): weight for calculating running
            exponential averages of gradient tensor mean and std values.
            Default: ``None`` 
    """

    def __init__(self, forward_bits, forward_avg_const,
                 backward_bits=None, backward_avg_const=None):
        super(Max, self).__init__(forward_bits, backward_bits)

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

        self.forward_max = None
        self.backward_max = None

        self.register_buffer('forward_bp', None)
        self.register_buffer('backward_bp', None)

    def forward(self, input):
        return MaxQuantFunction.apply(input, self)

    def update_forward_params(self, input):
        if not self.profiling:
            if self._buffers['forward_bp'] is None:
                raise ValueError('Forward quantization binary point (bp) is not defined.')
            return

        max = input.detach().abs().max()

        if self.forward_max is None:
            self.forward_max = max
        else:
            self.forward_max.mul_(1. - self.forward_avg_const).add_(max * self.forward_avg_const)

        bp = Max.binary_point(self.forward_max, self.forward_bits)

        if self._buffers['forward_bp'] is None:
            self._buffers['forward_bp'] = torch.tensor(1, dtype=torch.int, device=input.device)

        self._buffers['forward_bp'].fill_(bp)

        return

    def update_backward_params(self, input):
        if not self.profiling:
            if self._buffers['backward_bp'] is None:
                raise ValueError('Backward quantization binary point (bp) is not defined.')
            return

        max = input.detach().abs().max()

        if self.backward_max is None:
            self.backward_max = max
        else:
            self.backward_max.mul_(1. - self.backward_avg_const).add_(max * self.backward_avg_const)

        bp = Max.binary_point(self.backward_max, self.backward_bits)

        if self._buffers['backward_bp'] is None:
            self._buffers['backward_bp'] = torch.tensor(1, dtype=torch.int, device=input.device)

        self._buffers['backward_bp'].fill_(bp)

        return

    def forward_params(self):
        return self._buffers['forward_bp'].item(), self.forward_qmin, self.forward_qmax

    def backward_params(self):
        return self._buffers['backward_bp'].item(), self.backward_qmin, self.backward_qmax

    @staticmethod
    def binary_point(f_max, total_bits):
        M = math.ceil(math.log2(f_max + 1.0e-12))  # math.floor(math.log2(f_max)) + 1
        bp = total_bits - 1 - M
        return bp


class MaxQuantFunction(torch.autograd.Function):
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
        return MaxQuantFunction.quantize(input, bp, qmin, qmax)

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
        grad_input = MaxQuantFunction.quantize(grad_output, bp, qmin, qmax)

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