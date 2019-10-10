import torch
import math
from nn.quantization.quantmodule import QuantModule


class Agile(QuantModule):
    r"""Quantizes the tensor using the adaptive agile factor algorithm.

    The quantization is defined by the mapping between the real value,
    :math:`f`, and the quantized value, :math:`q` as:

    .. math::
        \begin{equation*}
        f = \Delta \times q,
        \end{equation*}

    where :math:`\Delta` is the scaling factor.

    In the agile factor algorithm, given the bit-width, :math:`W`, for
    quantization, :math:`\Delta` is determined from the agile max value,
    :math:`A_{\text{max}}`, of the tensor as:

    .. math::
        \begin{equation*}
        \Delta = 2^\left(W - 1 - ceil\left(\log_{2}A_{\text{max}}\right)\right),
        \end{equation*}

    where the agile max value, :math:`A_{\text{max}}` is determined as:

    .. math::
        \begin{equation*}
        A_{\text{max}} = k_{1}\left(\mu_{\text{max}} + k_{2}\sigma_{\text{max}}\right),
        \end{equation*}

    where :math:`\mu_{\text{max}}` and :math:`\sigma_{\text{max}}` are the mean
    and variance of the maximum tensor value, respectively, and :math:`k_{1}` and
    :math:`k_{2}` are the agile factor constants. For further
    details on :math:`\mu_{\text{max}}`, :math:`\sigma_{\text{max}}`,
    :math:`k_{1}`, and :math:`k_{2}` see reference.

    Args:
        forward_bits (int): bit-width of the forward tensor.
            Default: ``None`` (no quantization)
        backward_bits (int): bit-widht of the backward (gradient) tensor.
            Default: ``None`` (no quantization)

    Attributes:
        forward_max (float): :math:`A_{\text{max}}` agile max of the forward tensor.
        forward_mean (float): :math:`\mu_{\text{max}}` of the forward tensor.
        forward_var (float): :math:`\sigma_{\text{max}}` of the forward tensor.
        backward_max (float): :math:`A_{\text{max}}` agile max of the backward tensor.
        backward_mean (float): :math:`\mu_{\text{max}}` of the backward tensor.
        backward_var (float): :math:`\sigma_{\text{max}}` of the backward tensor.

    """
    _fudge_factor = 1.1
    _headroom = 0.1
    _avg_time_const = 0.125
    _S = _headroom * (1. / (1. - _avg_time_const)) * math.sqrt(math.pi / 2.)

    def __init__(self, forward_bits=None, backward_bits=None):
        super(Agile, self).__init__(forward_bits, backward_bits)

        self.forward_mean = None
        self.forward_var = None
        self.forward_max = None

        self.forward_delta = None
        self.forward_qmin = None
        self.forward_qmax = None

        self.backward_mean = None
        self.backward_var = None
        self.backward_max = None

        self.backward_delta = None
        self.backward_qmin = None
        self.backward_qmax = None

        if forward_bits:
            bound = math.pow(2.0, self.forward_bits - 1)
            self.forward_qmin = -bound
            self.forward_qmax = bound - 1

        if backward_bits:
            bound = math.pow(2.0, self.backward_bits - 1)
            self.backward_qmin = -bound
            self.backward_qmax = bound - 1

    def forward(self, input):
        '''
        '''
        return AgileFunction.apply(input, self)

    @staticmethod
    def agile_max(mean_max, var_max, size):
        headroom_factor = Agile._S * (1. - 0.0625 * math.log(size * 63.e-6))
        max_val = Agile._fudge_factor * (mean_max + headroom_factor * var_max)
        return max_val

    def forward_params(self):
        return self.forward_delta, self.forward_qmin, self.forward_qmax

    def backward_params(self):
        return self.backward_delta, self.backward_qmin, self.backward_qmax

    def update_forward_params(self, data):
        if not self.profiling:
            if self.forward_delta is None:
                raise ValueError('Forward quantization parameter delta is not defined.')
            return

        max_val = data.abs().max().item()

        if self.forward_mean is None or self.forward_var is None:
            # Initialize the mean and variance
            self.forward_mean = max_val
            self.forward_var = max_val
        else:
            # Update mean and variance
            beta = Agile._avg_time_const
            self.forward_mean = (1. - beta) * self.forward_mean + beta * max_val
            self.forward_var = (1. - beta) * self.forward_var + beta * abs(max_val - self.forward_mean)

        self.forward_max = Agile.agile_max(self.forward_mean, self.forward_var, data.numel())

        bp = self.forward_bits - (math.ceil(math.log2(self.forward_max + 1.e-12)) + 1)
        self.forward_delta = math.pow(2.0, -bp)

        return

    def update_backward_params(self, data):
        if not self.profiling:
            if self.backward_delta is None:
                raise ValueError('Backward quantization parameter delta is not defined.')
            return

        max_val = data.abs().max().item()

        if self.backward_mean is None or self.backward_var is None:
            # Initialize the mean and variance
            self.backward_mean = max_val
            self.backward_var = max_val
        else:
            # Update mean and variance
            beta = Agile._avg_time_const
            self.backward_mean = (1. - beta) * self.backward_mean + beta * max_val
            self.backward_var = (1. - beta) * self.backward_var + beta * abs(max_val - self.backward_mean)

        self.backward_max = Agile.agile_max(self.backward_mean, self.backward_var, data.numel())

        bp = self.backward_bits - (math.ceil(math.log2(self.backward_max + 1.e-12)) + 1)
        self.backward_delta = math.pow(2.0, -bp)

        return


class AgileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, module):
        '''
        '''
        ctx.module = module
        if not module.forward_bits:
            return input

        # Update forward quantization parameters if required
        module.update_forward_params(input)

        delta, low_bound, high_bound = module.forward_params()
        return AgileFunction.quantize(input, delta, low_bound, high_bound)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        '''
        module = ctx.module
        if not module.backward_bits:
            return grad_output, None

        # Update backward quantization parameters if required
        module.update_backward_params(grad_output)

        delta, low_bound, high_bound = module.backward_params()
        grad_input = AgileFunction.quantize(grad_output, delta, low_bound, high_bound)

        return grad_input, None

    @staticmethod
    def quantize(data, delta, low_bound, high_bound):
        output = data.div(delta).round()
        output.clamp_(low_bound, high_bound)
        output.mul_(delta)
        return output


# Modified SGD optimizer for Agile training
class AgileSGD(torch.optim.SGD):
    r'''SGD optimizer for Agile quantization-based training.

    Mean and variance of the tensor maximum are updated on the first batch
    of every epoch.

    Args:
        module (:class:`torch.nn.Module`): quantized model to train
        lr (float): learning rate
        device (:class:`torch.device`): the device to run training on
        batches_per_epoch (int): number of batches in each epoch

    Attributes:
        weight_updates (`dict`): dictionary of weight tensors to store weight
            updates. The precision of the `weight_updates` is the same as the
            weight gradient tensor.
        grad_quant_fn (`dict`): dictionary of quantization functions for
            quantizing the gradients of weights.

    Returns:
        None
    '''

    def __init__(self, module, lr, batches_per_epoch, device):
        super(AgileSGD, self).__init__(module.quant_net.parameters(), lr=lr)
        self.module = module
        self.weight_updates = {}
        self.grad_quant_fn = {}

        # Store the copy in weight_updates
        for tag, weight in self.module.quant_net.named_parameters():
            self.weight_updates[tag] = weight.data.to(device)

        self.batches_per_epoch = batches_per_epoch
        self.batch_num = 0

    def step(self, closure=None):
        self.batch_num += 1
        # Quantize weights
        for tag, weight in self.module.quant_net.named_parameters():
            quant_module = self.module.w_quant_modules[tag]
            if tag not in self.grad_quant_fn:
                # Get function objects for quantizing weight gradients
                self.weight_updates[tag].requires_grad_(True)
                quant_weight = quant_module(self.weight_updates[tag])
                self.grad_quant_fn[tag] = quant_weight.grad_fn
                self.weight_updates[tag].requires_grad_(False)
                weight.data = quant_weight.data
            else:
                weight.data = quant_module(self.weight_updates[tag]).data

        # Enable profiling (i.e. collect statistics) if first batch of the epoch
        if (self.batch_num % self.batches_per_epoch) == 1:
            self.module._profile(True)
        else:
            self.module._profile(False)

        loss = None
        if closure:
            loss = closure()

        # Learning rate schedule
        lr = self.param_groups[0]['lr']
        # Quantize gradients and add them to weight updates
        for tag, weight in self.module.quant_net.named_parameters():
            weight.grad.data, __ = self.grad_quant_fn[tag].apply(weight.grad.data)
            self.weight_updates[tag] = self.weight_updates[tag] - lr * weight.grad.data

        return loss
