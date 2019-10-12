import os
import torch.nn as nn
import torch
from tqdm import tqdm
from torchnet.engine import Engine
from easydict import EasyDict
import json
from utils.quantization_utils import *

def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return EasyDict(config_dict)


def expand_model(model):
    layers = []
    for layer in model.children():
        if len(list(layer.children())) > 0:
            layers += expand_model(layer)
        else:
            layers.append(layer)
    return layers


def count_nonzero_parameters(module):
    return torch.gt(torch.abs(module.weight.data), 0).sum().item()


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def test(model, testloader, loss_function, device):
    model.eval()
    model.to(device)

    engine = Engine()

    def compute_loss(data):
        inputs = data[0]
        labels = data[1]
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        return loss_function(outputs, labels), outputs

    def on_start(state):
        print("Running inference ...")
        state['iterator'] = tqdm(state['iterator'], leave=False)

    class Accuracy():
        _accuracy = 0.
        _sample_size = 0.

    def on_forward(state):
        batch_size = state['sample'][1].shape[0]
        Accuracy._sample_size += batch_size
        Accuracy._accuracy += batch_size * get_accuracy(state['output'].cpu(), state['sample'][1].cpu())

    engine.hooks['on_start'] = on_start
    engine.hooks['on_forward'] = on_forward

    engine.test(compute_loss, testloader)

    return Accuracy._accuracy / Accuracy._sample_size


def get_accuracy(outputs, labels):
    __, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()
    return accuracy


class ListAverageMeter(object):
    """Computes and stores the average and current values of a list"""
    def __init__(self):
        self.len = 10000  # set up the maximum length
        self.reset()

    def reset(self):
        self.val = [0] * self.len
        self.avg = [0] * self.len
        self.sum = [0] * self.len
        self.count = 0

    def set_len(self, n):
        self.len = n
        self.reset()

    def update(self, vals, n=1):
        assert len(vals) == self.len, 'length of vals not equal to self.len'
        self.val = vals
        for i in range(self.len):
            self.sum[i] += self.val[i] * n
        self.count += n
        for i in range(self.len):
            self.avg[i] = self.sum[i] / self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConvLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=None):
        super(ConvLayerRotation, self).__init__()
        self.bias = bias
        cout, cin = rotation_matrix.size()
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3)
        self.conv = nn.Conv2d(out_channels=cout, in_channels=cin, kernel_size=1, padding=0, stride=1, bias=False)
        self.conv.weight.data = rotation_matrix

    def forward(self, x):
        if self.bias is not None:
            x = torch.cat([x, self.bias*x.new_ones(x.size(0), 1, x.size(2), x.size(3))], 1)
        return self.conv(x)


class LinearLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=None):
        super(LinearLayerRotation, self).__init__()
        self.bias = bias
        cout, cin = rotation_matrix.size()
        self.linear = nn.Linear(in_features=cin, out_features=cout, bias=False)
        self.linear.weight.data = rotation_matrix

    def forward(self, x):
        if self.bias is not None:
            x = torch.cat([x, self.bias*x.new_ones(x.size(0), 1)], 1)
        return self.linear(x)


class EigenBasisLayer(nn.Module):
    def __init__(self, Q_G, Q_A, M_new_basis, module, use_bias):
        super(EigenBasisLayer, self).__init__()
        self.sequential = update_layer_basis(module, Q_G, Q_A, M_new_basis, use_bias)

    def forward(self, x):
        return self.sequential(x)


class OriginalBasisLayer(nn.Module):
    def __init__(self, module):
        super(OriginalBasisLayer, self).__init__()
        self.sequential = nn.Sequential(module)

    def forward(self, x):
        return self.sequential(x)


class BasisLayer(nn.Module):
    def __init__(self, module):
        super(BasisLayer, self).__init__()
        self.basis = OriginalBasisLayer(module)

    def forward(self, x):
        return self.basis(x)


def update_layer_basis(module, Q_G, Q_A, M_new_basis, use_bias):
    if isinstance(module, nn.Conv2d):
        patch_size = M_new_basis.size(1)
        if use_bias:
            bias = 1/patch_size
        else:
            bias = None
        rotation_conv_A = ConvLayerRotation(Q_A.t(), bias=bias)
        rotation_conv_G = ConvLayerRotation(Q_G)
        M_new_basis = M_new_basis.view(M_new_basis.size(0), -1, module.kernel_size[0], module.kernel_size[1])
        cout, cin, kh, kw = M_new_basis.size()
        conv_new_basis = nn.Conv2d(out_channels=cout, in_channels=cin, kernel_size=(kh, kw), stride=module.stride,
                                   padding=module.padding, bias=False)
        conv_new_basis.weight.data = M_new_basis
        return nn.Sequential(
            rotation_conv_A,
            conv_new_basis,
            rotation_conv_G
        )
    elif isinstance(module, nn.Linear):
        if use_bias:
            bias = 1
        else:
            bias = None
        rotation_linear_A = LinearLayerRotation(Q_A.t(), bias=bias)
        rotation_linear_G = LinearLayerRotation(Q_G)
        cout, cin = M_new_basis.size()
        linear_new_basis = nn.Linear(out_features=cout, in_features=cin, bias=False)
        linear_new_basis.weight.data = M_new_basis
        return nn.Sequential(
            rotation_linear_A,
            linear_new_basis,
            rotation_linear_G
        )
    else:
        raise NotImplementedError


def load_checkpoint_pruning(checkpoint_path, net, use_bias):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.add_basis()
    shapes = []
    for module_name in list(checkpoint.keys()):
        if 'sequential' in module_name:
            shapes.append(checkpoint[module_name].shape)
    to_delete = []
    for module in net.modules():
        if isinstance(module, BasisLayer):
            main_module = module.basis.sequential[0]
            use_bias_module = use_bias & (main_module.bias is not None)
            if isinstance(main_module, nn.Conv2d):
                cout, cin, kh, kw = main_module.weight.shape
                Q_G = torch.rand(cout, cout)
                Q_A = torch.rand(cin, cin)
                M_new_basis = torch.rand(cout, cin, kh, kw)
            else:
                cout, cin = main_module.weight.shape
                Q_G = torch.rand(cout, cout)
                Q_A = torch.rand(cin, cin)
                M_new_basis = torch.rand(cout, cin)
            new_basis_layer = EigenBasisLayer(Q_G, Q_A, M_new_basis, main_module, use_bias=use_bias_module)
            to_delete.append(module.basis)
            module.basis = new_basis_layer
    for m in to_delete:
        m.cpu()
        del m

    interesting_modules = []
    for module in net.modules():
        if isinstance(module, EigenBasisLayer):
            main_module = module.sequential[1]
            if isinstance(main_module, nn.Conv2d):
                rotation_A = module.sequential[0].conv
                rotation_G = module.sequential[2].conv
            else:
                rotation_A = module.sequential[0].linear
                rotation_G = module.sequential[2].linear
            interesting_modules += [rotation_A, main_module, rotation_G]

    ct = 0
    for module in interesting_modules:
        module.weight.data = torch.rand(shapes[ct])
        if isinstance(module, nn.Conv2d):
            module.out_channels = shapes[ct][0]
            module.in_channels = shapes[ct][1]
        else:
            module.out_features = shapes[ct][0]
            module.in_features = shapes[ct][1]
        ct += 1
    net.load_state_dict(checkpoint)
    return net


def fuse_conv_bnorm(conv, bn):
    with torch.no_grad():
        # init
        # fusedconv = torch.nn.Conv2d(
        conv_class = type(conv)
        fusedconv = conv_class(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True,
            groups=conv.groups
        )

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        if conv.bias is not None:
            b_conv = conv.bias
        else:
            b_conv = torch.zeros(conv.weight.size(0))
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(b_conv + b_bn)

        return fusedconv


def replace_bn(model):
    for k, v in model._modules.items():
        if len(v._modules.items()) > 0:
            replace_bn(v)
        else:
            if not isinstance(v, EmptyLayer):
                if isinstance(v, nn.BatchNorm2d):
                    if not hasattr(v, 'no_merging'):
                        model._modules[k] = EmptyLayer()


def remove_bn(net):
    last_conv = None
    l = []
    for module in net.modules():
        if isinstance(module, EigenBasisLayer):
            main_module = module.sequential[1]
            if isinstance(main_module, nn.Conv2d):
                last_conv = module.sequential[2].conv
        if isinstance(module, nn.BatchNorm2d):
            if not hasattr(module, 'no_merging'):
                l.append(fuse_conv_bnorm(last_conv, module))

    ct = 0
    for module in net.modules():
        if isinstance(module, EigenBasisLayer):
            main_module = module.sequential[1]
            if isinstance(main_module, nn.Conv2d):
                module.sequential[2] = l[ct]
        if isinstance(module, nn.BatchNorm2d):
            if not hasattr(module, 'no_merging'):
                ct += 1

    replace_bn(net)
