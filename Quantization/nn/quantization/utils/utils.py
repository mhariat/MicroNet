import copy
import torch
import torch.nn
from collections import OrderedDict
from torchnet.engine import Engine
from tqdm import tqdm
from utils.utils import get_accuracy
from nn import EmptyLayer


def stochastic_round(x):
    '''Rounds x stochastically'''
    x_sign = x.sign()
    x_abs = x.abs()
    x_int = x_abs.floor()
    x_diff = x_abs - x_int

    rand_float = x.new(x.shape)
    rand_float.uniform_(0., 1.)

    rand_sign = (x_diff - rand_float).sign()

    x_round = x_int + 0.5 * (rand_sign + 1)
    return x_sign * x_round


def insert_quant_modules(model, a_quant_module):
    r'''Inserts quantization modules at the ouput of each module/layer
    in a given graph.
    '''
    quant_modules = []
    for k, v in model._modules.items():
        # If module has children, recursively attach quant modules to the sub-modules of the module.
        if len(v._modules.items()) > 0:
            modules, __ = insert_quant_modules(v, a_quant_module)
            for module in modules:
                quant_modules.append(module)
        else:
            if not isinstance(v, EmptyLayer):
                new_modules = OrderedDict()
                # keep the original module
                new_modules[k] = v
                # add the quantization module
                new_quant_module = copy.deepcopy(a_quant_module)
                new_modules['quant'] = new_quant_module
                quant_modules.append(new_quant_module)
                # replace the module with a sequential module containing the original model and quantization layer(s)
                model._modules[k] = torch.nn.Sequential(new_modules)

    return quant_modules, model


def calibrate(model, testloader, loss_function, device):
    """Calibrates the weight and activation quantization parameters.

    Executes forward passes using the input data from `testloader`.
    For every forward pass, collects the statistics and calibrates
    the quantization parameters for all the weight and activation
    quant modules.

    Arguments:
        model (:class:`QuantizedNet`): nn model to train
        testloader (:class:`torch.utils.data.DataLoader`): dataloader to
            iterate through the sample data for calibration
        loss_function (:class:`torch.nn._Loss`): function to compute loss
        device (:class:`torch.device`): the device to run calibration on

    Returns:
        Module: calibrated quantized model
    """
    model.eval()
    model.to(device)

    engine = Engine()

    # Enable profiling to collect statistics and calibrate quant params
    model._profile()

    # Quantize weights
    model._quantize_weights()

    def compute_loss(data):
        """Computes the loss from a given nn model."""
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        return loss_function(outputs, labels), outputs

    def on_start(state):
        print("Calibrating quantized network ...")
        state['iterator'] = tqdm(state['iterator'], leave=False)

    def on_forward(state):
        loss = state['loss'].item()
        accuracy = get_accuracy(state['output'].cpu(), state['sample'][1].cpu())
        state['iterator'].write('batch %d loss %.3f accuracy %.3f ' % (state['t'], loss, accuracy), end='\n')

    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward

    engine.test(compute_loss, testloader)

    return model
