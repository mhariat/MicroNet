import torch
import copy
from torch.nn.parameter import Parameter
from nn.quantization.utils.utils import insert_quant_modules


class QuantizedNet(torch.nn.Module):
    """Class for the quantized neural network.

    QuantizedNet defines the structure of the quantized nn and an interface
    to perform forward and backward passes with quantized weights and
    activations.

    It takes a nn model, adds quantization modules for the weights and
    inserts quantization layers after each activation layer in the nn model.

    To quantize pre-trained network for quantized inference, we need to
    calibrate the quantization parameters for the weights and activations.
    This can be achieved by passing the quantized model and calibration input
    data to the function `wavetorch.quant.utils.calibrate`.

    To train the `QuantizedNet` nn, follow the procedure similar to training
    any other PyTorch nn model. You can use WaveTorch's utility function
    `wavetorch.utils.train` for training `QuantizedNet` nn.

    There are three modes of `QuantizedNet`: eval, training, and profiling.
    eval and training are the usual modes in `nn.Module`. profiling mode is
    for setting the flag to collect the quantization statistics, such as
    running means/maxs of activation/weight tensors, while calibrating
    pre-trained network for inference or peforming quantized training.

    Arguments:
        net (class:`nn.Module`): a neural network model to quantize
        a_quant_module (class:`QuantModule`): quantization module to
            quantize activations
        w_quant_module (class:`QuantModule`): quantization module to
            quantize weights

    Returns:
        QuantizedNet: a quantized neural network model

    Here is a small example::

        # Define the target nn model to quantize
        model = nn.Sequential(OrderedDict([
                    ('conv1', nn.Conv2d(1,20,5)),
                    ('relu1', nn.ReLU()),
                    ('conv2', nn.Conv2d(20,64,5)),
                    ('relu2', nn.ReLU()),
                    ('fc1', nn.Linear(1024,10))
                    ]))

        # Define quantization schemes for activations and weights
        a_quant = wavetorch.quant.Uniform(forward_bits=8, backward_bits=8)
        w_quant = wavetorch.quant.Uniform(forward_bits=8, backward_bits=8)

        # Get quantized model
        quant_model = wavetorch.quant.QuantizedNet(model, a_quant, w_quant)

    """

    def __init__(self, net, a_quant_module, w_quant_module):
        super(QuantizedNet, self).__init__()

        self.quant_input = copy.deepcopy(a_quant_module)

        # Insert activation quant modules at the output of each module in a given network
        # and store the list of added activation quant modules
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
        """Overwrites the base class torch.nn.Module's `train` function.

        Here it is used to enable quantization profiling when training.

        Return:
            Module: self
        """
        super(QuantizedNet, self).train(mode)
        self._profile(mode)

        return self

    def eval(self):
        """Sets the module in evaluation mode.

        Disables training and profiling.
        """
        return self.train(False)

    def to(self, *args, **kwargs):
        """Overwrites the base class torch.nn.Module's `to` function.

        Here it is used to move weight quant modules to the device.
        """
        super(QuantizedNet, self).to(*args, **kwargs)

        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        # Move weight quant layers to device
        if device is not None:
            for name, param in self.quant_net.named_parameters():
                quant_module = self.w_quant_modules[name]
                quant_module.to(device)

    def _quantize_weights(self):
        # Quantize weights
        for name, param in self.quant_net.named_parameters():
            quant_module = self.w_quant_modules[name]
            if 'conv' not in name:
                with torch.no_grad():
                    param.data = quant_module(param.data)
            elif 'conv' in name:
                # if param.shape[0]*param.shape[2]*param.shape[3] != 1:
                with torch.no_grad():
                    param.data = quant_module(param.data)
                # mean = self.w_quant_modules[name].forward_mean
                # std = self.w_quant_modules[name].forward_std
                # max = self.w_quant_modules[name].forward_max
                # min = self.w_quant_modules[name].forward_min
                # print(param.shape, mean, std, max, min)

    def _quantize_grad_weights(self):
        for name, param in self.quant_net.named_parameters():
            if param.grad is None:
                continue
            quant_module = self.w_quant_modules[name]
            with torch.no_grad():
                param.grad.data = quant_module.backward(param.grad.data)

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
