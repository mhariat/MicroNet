import torch
from nn.quantization.quantmodule import QuantModule


class Uniform(QuantModule):
    r"""Quantizes the tensor using the uniform quantization scheme as described
    in [Banner2018]_.

    The quantization is defined by the mapping between the real value,
    :math:`f`, and the quantized value, :math:`q` as an affine map [GEMMLOWP]_:

    .. math::
        \begin{equation*}
        f = \Delta\left(q - q_0\right),
        \end{equation*}

    where :math:`\Delta` is the scaling factor and :math:`q_0` is the zero-point,
    i.e., :math:`f = 0` when :math:`q = q_0`.

    Given a float tensor, :math:`T_{\text{f}}`, the quantized tensor,
    :math:`T_{\text{q}}`, using :math:`W` bits is computed as:

    .. math::
        \begin{equation*}
        \Delta = \frac{\left(\overline{v_{\text{max}}} - \overline{v_{\text{min}}}\right)}{2^{W}-1},
        \end{equation*}

    .. math::
        \begin{equation*}
        q_0 = \text{round}\left(\min\left(\max\left(-\frac{\overline{v_{\text{min}}}}{\Delta},0\right),2^W - 1\right)\right),
        \end{equation*}

    .. math::
        \begin{equation*}
        T_{\text{q}} = \text{round}\left(\frac{T_{\text{f}}}{\Delta} + q_0\right),
        \end{equation*}

    where :math:`\overline{v_{\text{max}}}` and :math:`\overline{v_{\text{min}}}`
    are the mean (over different batches) maximum and minimum values of the tensor,
    :math:`T_{\text{f}}`, respectively. :math:`\overline{v_{\text{max}}}`
    and :math:`\overline{v_{\text{min}}}` are computed using the exponential
    averaging scheme.

    Args:
        forward_bits (int): bit-width of the forward tensor
        forward_avg_const (float): weight for calculating running
            exponential averages of forward pass tensor min and max values
        backward_bits (int): bit-widht of the backward (gradient) tensor.
            Default: ``None``
        backward_avg_const (float): weight for calculating running
            exponential averages of gradient tensor min and max values
            Default: ``None``
        batch_average (bool): set whether to compute max and min values 
            as an average over batches in a given tensor. Default: ``None``


    .. [Banner2018] `"Scalable methods for 8-bit training of neural networks", Banner R. et al., arXiv, 2018`__.

    __ http://arxiv.org/abs/1805.11046

    .. [GEMMLOWP] https://github.com/google/gemmlowp/blob/master/doc/quantization.md

    """

    def __init__(self, forward_bits, forward_avg_const,
                 backward_bits=None, backward_avg_const=None,
                 batch_average=False):
        super(Uniform, self).__init__(forward_bits, backward_bits)

        self.batch_average = batch_average

        self.forward_avg_const = forward_avg_const
        self.backward_avg_const = backward_avg_const

        self.forward_qmin = None
        self.forward_qmax = None
        if self.forward_bits:
            self.forward_qmin = 0
            self.forward_qmax = 2 ** self.forward_bits - 1

        self.backward_qmin = None
        self.backward_qmax = None
        if self.backward_bits:
            self.backward_qmin = 0
            self.backward_qmax = 2 ** self.backward_bits - 1

        self.forward_vmin = None
        self.forward_vmax = None

        self.backward_vmin = None
        self.backward_vmax = None

        self.register_buffer('forward_delta', None)
        self.register_buffer('backward_delta', None)

        self.register_buffer('forward_q0', None)
        self.register_buffer('backward_q0', None)

    def forward(self, input):
        return UniformFunction.apply(input, self)

    def update_forward_params(self, input):
        if not self.profiling:
            if self._buffers['forward_delta'] is None or self._buffers['forward_q0'] is None:
                raise ValueError('Forward quantization parameters delta and q0 not defined.')
            return

        batch_size = max(1, self.batch_average * input.size(0))
        min_value = input.detach().view(batch_size, -1).min(-1)[0].mean()
        max_value = input.detach().view(batch_size, -1).max(-1)[0].mean()

        if self.forward_vmin is None or self.forward_vmax is None:
            self.forward_vmin = min_value
            self.forward_vmax = max_value
        else:
            self.forward_vmin.mul_(1. - self.forward_avg_const).add_(min_value * self.forward_avg_const)
            self.forward_vmax.mul_(1. - self.forward_avg_const).add_(max_value * self.forward_avg_const)

        delta, q0 = Uniform.affinemap(self.forward_vmax, self.forward_vmin, self.forward_qmax, self.forward_qmin)

        if self._buffers['forward_delta'] is None:
            self._buffers['forward_delta'] = torch.tensor(1, dtype=torch.float, device=input.device)
        if self._buffers['forward_q0'] is None:
            self._buffers['forward_q0'] = torch.tensor(1, dtype=torch.float, device=input.device)

        self._buffers['forward_delta'].fill_(delta)
        self._buffers['forward_q0'].fill_(q0)

        return

    def update_backward_params(self, input):
        if not self.profiling:
            if self._buffers['backward_delta'] is None or self._buffers['backward_q0'] is None:
                raise ValueError('Backward quantization parameters delta and q0 not defined.')
            return

        batch_size = max(1, self.batch_average * input.size(0))
        min_value = input.detach().view(batch_size, -1).min(-1)[0].mean()
        max_value = input.detach().view(batch_size, -1).max(-1)[0].mean()

        if self.backward_vmin is None or self.backward_vmax is None:
            self.backward_vmin = min_value
            self.backward_vmax = max_value
        else:
            self.backward_vmin.mul_(1. - self.backward_avg_const).add_(min_value * self.backward_avg_const)
            self.backward_vmax.mul_(1. - self.backward_avg_const).add_(max_value * self.backward_avg_const)

        delta, q0 = Uniform.affinemap(self.backward_vmax, self.backward_vmin, self.backward_qmax, self.backward_qmin)

        if self._buffers['backward_delta'] is None:
            self._buffers['backward_delta'] = torch.tensor(1, dtype=torch.float, device=input.device)
        if self._buffers['backward_q0'] is None:
            self._buffers['backward_q0'] = torch.tensor(1, dtype=torch.float, device=input.device)

        self._buffers['backward_delta'].fill_(delta)
        self._buffers['backward_q0'].fill_(q0)

        return

    def forward_params(self):
        return self._buffers['forward_delta'].item(), self._buffers[
            'forward_q0'].item(), self.forward_qmin, self.forward_qmax

    def backward_params(self):
        return self._buffers['backward_delta'].item(), self._buffers[
            'backward_q0'].item(), self.backward_qmin, self.backward_qmax

    @staticmethod
    def affinemap(vmax, vmin, qmax, qmin):
        delta = (vmax - vmin) / (qmax - qmin)
        delta = max(delta, 1.e-12)
        q0 = int(min(max(-vmin / delta, 0), qmax))
        return delta, q0


class UniformFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, module):
        '''
        '''
        ctx.module = module
        if not module.forward_bits:
            return input

        # Update forward quantization parameters if required
        module.update_forward_params(input)

        delta, q0, qmin, qmax = module.forward_params()
        return UniformFunction.quantize(input, delta, q0, qmin, qmax)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        '''
        module = ctx.module
        if not module.backward_bits:
            return grad_output, None

        # Update backward quantization parameters if required
        module.update_backward_params(grad_output)

        delta, q0, qmin, qmax = module.backward_params()
        grad_input = UniformFunction.quantize(grad_output, delta, q0, qmin, qmax)

        return grad_input, None

    @staticmethod
    def quantize(input, delta, q0, qmin, qmax):
        # Quantize
        output = input.div(delta).add(q0)
        output.clamp_(qmin, qmax).round_()
        # Dequantize
        output.sub_(q0).mul_(delta)
        return output

    @staticmethod
    def symbolic(g, input, module):
        '''
        '''
        bp, qmin, qmax = module.forward_params()
        kwargs = {"versipoint_i": [module.forward_bits, bp]}
        quant_node = g.op("Quant", input, **kwargs)

        return quant_node
