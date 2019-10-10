import torch
from nn.quantization.quantmodule import QuantModule
from nn.quantization.utils.utils import stochastic_round


class Wage(QuantModule):
    r"""Quantizes the tensor using the WAGE quantization scheme [Wu2018]_.

    In WAGE scheme, the mapping between the float value, :math:`f`, and
    the quantized value, :math:`q` is defined as:

    .. math::
        \begin{equation*}
        q = \text{round}\left(\frac{f}{\alpha\Delta}\right),
        \end{equation*}

    .. math::
        \begin{equation*}
        f = \text{clip}\left[\Delta \times q,-1+\Delta, 1-\Delta\right],
        \end{equation*}

    where, :math:`\Delta = 2^{1-W}` for :math:`W` bit-width and :math:`\alpha`
    is the shift-factor to scale :math:`f` before quantization.

    Args:
        forward_bits (int): bit-width of the forward tensor. Default: ``None``
            (no quantization)
        forward_shift (`function`): shift operation to apply before quantizing
            tensor in forward pass. Default: ``None``
        forward_stochastic (bool): flag to use stochastic rounding for quantizing
            tensor in forward pass. Default: ``False``
        backward_bits (int): bit-width of the backward tensor. Default: ``None``
            (no quantization)
        backward_shift (`function`): shift operation to apply before quantizing
            tensor in backward pass. Default: ``None``
        backward_stochastic (bool): flag to use stochastic rounding for quantizing
            tensor in backward pass. Default: ``False``

    Attributes:
        forward_delta (float): :math:`\Delta` for the forward quantization.
        forward_min (float): lower bound of the quantized range in forward pass.
        forward_max (float): upper bound of the quantized range in forward pass.
        backward_delta (float): :math:`\Delta` for the backward quantization.
        backward_min (float): lower bound of the quantized range in backward pass.
        backward_max (float): upper bound of the quantized range in backward pass.

    .. [Wu2018] `"Training and inference with integers in deep neural networks", Wu S. et al., ICLR, 2018`__.

    __ https://arxiv.org/abs/1802.04680

    """

    def __init__(self, forward_bits=None, forward_shift=None, forward_stochastic=False,
                 backward_bits=None, backward_shift=None, backward_stochastic=False):
        super(Wage, self).__init__(forward_bits, backward_bits)

        self.forward_shift = forward_shift
        self.backward_shift = backward_shift
        self.forward_stochastic = forward_stochastic
        self.backward_stochastic = backward_stochastic

        if forward_bits:
            self.forward_bp = forward_bits - 1
            self.forward_delta = 2.0 ** (-self.forward_bp)
            self.forward_min = -1.0 + self.forward_delta
            self.forward_max = 1.0 - self.forward_delta
        else:
            self.forward_bp = None
            self.forward_delta = None
            self.forward_min = None
            self.forward_max = None

        if backward_bits:
            self.backward_bp = backward_bits - 1
            self.backward_delta = 2.0 ** (-self.backward_bp)
            self.backward_min = -1.0 + self.backward_delta
            self.backward_max = 1.0 - self.backward_delta
        else:
            self.backward_bp = None
            self.backward_delta = None
            self.backward_min = None
            self.backward_max = None

    def forward(self, input):
        '''

        '''
        return WageFunction.apply(input, self)


class WageFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, module):
        ctx.module = module
        output = input
        if not module.forward_bits:
            return output

        if module.forward_shift:
            output = module.forward_shift(output)

        mask = (output.ge(module.forward_min) + output.le(module.forward_max)).ge(2)

        ctx.save_for_backward(mask)

        output.clamp_(module.forward_min, module.forward_max)

        output = output.div(module.forward_delta)

        if module.forward_stochastic:
            output = stochastic_round(output)
        else:
            output = output.round()

        output.mul_(module.forward_delta)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        '''

        '''
        module = ctx.module
        mask = ctx.saved_tensors[0]

        grad_input = grad_output
        if not module.backward_bits:
            return grad_input, None

        if module.backward_shift:
            grad_input = module.backward_shift(grad_input)

        grad_input.clamp_(module.backward_min, module.backward_max)

        grad_input = grad_input.div(module.backward_delta)

        if module.backward_stochastic:
            grad_input = stochastic_round(grad_input)
        else:
            grad_input = grad_input.round()

        grad_input.mul_(module.backward_delta)

        grad_input.mul_(mask.float())

        return grad_input, None


class WageSGD(torch.optim.SGD):
    r'''SGD optimizer for WAGE quantization-based training.

    To update and quantize weights, WageSGD implements the same scheme as in
    the reference `TensorFlow implementation of WAGE
    <https://github.com/boluoweifenda/WAGE>`_.

    In WageSGD, we stores a local copy of the weights in :math:`\tilde{W}`,
    which has the same precision as the gradients. :math:`\tilde{W}` copy
    allows us to accumulate small weight updates which would otherwise be lost
    in weight quantization process. For example, the if the step size,
    :math:`dW_i`, for updating weight :math:`W_i` is less than half of the
    quantization step, :math:`\Delta`, i.e.
    :math:`\left|dW_i\right|<\frac{\Delta}{2}`, then the quantized value after
    the weight update will not change, i.e., :math:`Q(W_i-dW_i) = W_i`.

    The weights are updated and quantized in the following manner:

    - Get quantized weights by quantizing :math:`\tilde{W}`.
    - Compute gradients.
    - Quantize gradients using Eq. 10 and 11 in [Wu2018]_ as:

    .. math::
        \begin{equation*}
        s_{\text{G}} = 2^{\text{round}\left(\log_2\left(\max(\left|dW_{\text{f}}\right|)\right)\right)}
        \end{equation*}

    .. math::
        \begin{equation*}
         dW_{\text{s}} = \eta\times\frac{dW_{\text{f}}}{s_{\text{G}}},
        \end{equation*}

    .. math::
        \begin{equation*}
         dW_{\text{q}} = \Delta_{\text{G}}\times\text{sign}\times\left(dW_{\text{s}}\right)\left(\lfloor\left|dW_{\text{s}}\right|\rfloor + \text{Bernoulli}\left(\left|dW_{\text{s}}\right|\lfloor\left|dW_{\text{s}}\right|\rfloor\right)\right),
        \end{equation*}

    where, :math:`dW_{\text{q}}` is the quantized gradient,
    :math:`dW_{\text{f}}` is the gradient in float, :math:`dW_{\text{s}}` is
    the shifted gradient, :math:`s_{\text{G}}` is the shift value,
    :math:`\Delta_{G}` is the quantization step size for the gradient,and
    :math:`\text{Bernoulli}` is the function to stochastically sample decimal
    part to either 0 or 1.

    - Accumulate :math:`dW_{\text{q}}` in :math:`\tilde{W}` as:

    .. math::
        \begin{equation*}
        \tilde{W} = \tilde{W} - dW_{\text{q}}
        \end{equation*}

    Args:
        module (:class:`torch.nn.Module`): quantized model to train.
        weight_shifts (`dict`): shift values to scale weights after
            quantization see Eq. 7 in [Wu2018]_.
        lr (float): learning rate.
        device (:class:`torch.device`): the device to run training on.

    Attributes:
        weight_updates (`dict`): dictionary of weight tensors to store weight
            updates. The precision of the `weight_updates` is the same as the
            weight gradient tensor.
        grad_quant_fn (`dict`): dictionary of quantization functions for
            quantizing the gradients of weights.

    Returns:
        None
    '''

    def __init__(self, module, weight_shifts, lr, device):
        super(WageSGD, self).__init__(module.quant_net.parameters(), lr=lr)
        self.module = module
        self.weight_shifts = weight_shifts
        self.weight_updates = {}

        # Store the copy in weight_updates
        for tag, weight in self.module.quant_net.named_parameters():
            self.weight_updates[tag] = weight.data.to(device)

    def step(self, closure=None):
        # Quantize weights
        for tag, weight in self.module.quant_net.named_parameters():
            quant_module = self.module.w_quant_modules[tag]
            weight.data = quant_module(self.weight_updates[tag]).data.div(self.weight_shifts[tag])

        # Backprop/compute gradients
        loss = None
        if closure:
            loss = closure()

        # Learning rate schedule
        lr = self.param_groups[0]['lr']
        # Quantize gradients and add them to weight updates
        for tag, weight in self.module.quant_net.named_parameters():
            quant_module = self.module.w_quant_modules[tag]
            w_bits = quant_module.forward_bits
            grad_bits = quant_module.backward_bits

            if grad_bits:
                # Quantize gradient
                if quant_module.backward_shift:
                    weight.grad.data = quant_module.backward_shift(weight.grad.data)

                weight.grad.data = WageSGD.quantize_grad(weight.grad.data, grad_bits, quant_module.backward_stochastic,
                                                         lr)
                self.weight_updates[tag] = self.weight_updates[tag] - weight.grad.data
            else:
                self.weight_updates[tag] = self.weight_updates[tag] - lr * weight.grad.data

            if w_bits:
                # Clip updated weights
                self.weight_updates[tag] = self.weight_updates[tag].clamp(quant_module.forward_min,
                                                                          quant_module.forward_max)

        return loss

    @staticmethod
    def quantize_grad(grad, bit_width, stochastic, lr):
        grad_quant = grad.mul(lr)
        if stochastic:
            grad_quant = stochastic_round(grad_quant)
        else:
            grad_quant.round_()
        grad_quant.mul_(2 ** (1 - bit_width))
        return grad_quant
