import torch


class QuantModule(torch.nn.Module):
    """Base class for all quantization modules.

    QuantModule defines the interface for quantizing the nn tensors,
    such as weights, activations, gradients of activations, and
    gradients of weights.

    It allows to define different bit-widths (precision) for the forward
    pass tensor and backward pass tensor, i.e., gradients.

    .. note::
        By default, the gradient tensor is assumed to be in full precision
        (e.g. float32). To quantize the gradient tensor, specify bit-width 
        for the backward pass as `backward_bits` argument.

    .. note::
        Assumes signed integer representation using given number of bits.

    .. note::
        `profiling` flag is used to set the mode to collect quantization statistics
        in the forward and backward passes.

    Arguments:
        forward_bits (int): bit-width of the forward tensor
        backward_bits (int): bit-widht of the backward (gradient) tensor.
            Default: ``None``

    """

    def __init__(self, forward_bits=None, backward_bits=None):
        super(QuantModule, self).__init__()
        self.forward_bits = forward_bits
        self.backward_bits = backward_bits

        self.profiling = False
        self.quant_function = None

    def forward(self, *input):
        r""" Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def profile(self, mode=True):
        r"""Sets the module in profiling mode.

        In the profiling mode, the module collects various statistics it needs to 
        compute the quantization parameters, such as scaling factor.

        Returns:
            Module: self
        """
        self.profiling = mode
        return self
