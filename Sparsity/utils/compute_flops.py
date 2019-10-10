import torch
import torch.nn as nn
from collections import OrderedDict
import gc


def compute_conv2d_flops(module, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()

    kernel_ops_mul = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels / module.groups)
    kernel_ops_add = kernel_ops_mul - 1
    bias_ops = 1 if module.bias is not None else 0

    flops_add = (kernel_ops_add + bias_ops) * output_channels * output_height * output_width * batch_size
    flops_mul = kernel_ops_mul * output_channels * output_height * output_width * batch_size
    return flops_mul + flops_add


def compute_linear_flops(module, input, output):
    batch_size = input[0].size(0) if input[0].dim() == 2 else 1
    weight_ops_mul = module.weight.size(1)
    weight_ops_add = weight_ops_mul - 1
    bias_ops = 1 if module.bias is not None else 0
    flops = batch_size * module.weight.size(0) * (weight_ops_add + weight_ops_mul + bias_ops)
    return flops


def compute_batchnorm2d_flops(module, input, output):
    return input[0].nelement()*2


def compute_relu_flops(module, input, output):
    return input[0].nelement()


def compute_maxpool2d_flops(module, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()
    kernel_ops = module.kernel_size * module.kernel_size - 1
    flops = kernel_ops * output_channels * output_height * output_width * batch_size
    return flops


def compute_avgpool2d_flops(module, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()
    if isinstance(module.kernel_size, tuple):
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] - 1
    else:
        kernel_ops = module.kernel_size * module.kernel_size - 1
    kernel_avg = 1
    flops = (kernel_ops + kernel_avg) * output_channels * output_height * output_width * batch_size
    return flops


def compute_softmax_flops(module, input, output):
    count = input[0].nelement()
    exp = count
    add = count - 1
    div = count
    return exp + add + div


def compute_sigmoid_flops(module, input, output):
    count = input[0].nelement()
    exp = 1
    add = 1
    div = 1
    return count * (exp + add + div)


def compute_lstm_flops(module, input, output):
    batch_size = input[0].size(0) if input[0].dim() == 2 else 1
    embedding_size = module.input_size
    hidden_size = module.hidden_size
    weight_ops_mul = embedding_size
    weight_ops_add = weight_ops_mul - 1
    bias_ops = 1
    linear_flops = hidden_size * (weight_ops_add + weight_ops_mul + bias_ops)
    activation_flops = hidden_size
    element_wise_multi_flops = hidden_size
    element_wise_add_flops = hidden_size

    lstm_flops = batch_size * (8*linear_flops + 9*activation_flops + 3*element_wise_multi_flops +
                               element_wise_add_flops)

    return lstm_flops


def get_total_flops(model, input_res, batch_size):
    cuda = torch.cuda.is_available()

    list_conv = []

    def conv_hook(self, input, output):
        res = compute_conv2d_flops(self, input, output)
        list_conv.append(res)

    list_linear = []

    def linear_hook(self, input, output):
        res = compute_linear_flops(self, input, output)
        list_linear.append(res)

    list_bn = []

    def bn_hook(self, input, output):
        res = compute_batchnorm2d_flops(self, input, output)
        list_bn.append(res)

    list_relu = []

    def relu_hook(self, input, output):
        res = compute_relu_flops(self, input, output)
        list_relu.append(res)

    def relu6_hook(self, input, output):
        res = 2*compute_relu_flops(self, input, output)
        list_relu.append(res)
        # TODO: should I multiply by 2?

    list_pooling = []

    def max_pooling_hook(self, input, output):
        res = compute_maxpool2d_flops(self, input, output)
        list_pooling.append(res)

    def avg_pooling_hook(self, input, output):
        res = compute_avgpool2d_flops(self, input, output)
        list_pooling.append(res)

    def softmax_pooling_hook(self, input, output):
        res = compute_softmax_flops(self, input, output)
        list_pooling.append(res)

    def sigmoid_pooling_hook(self, input, output):
        res = compute_sigmoid_flops(self, input, output)
        list_pooling.append(res)

    list_lstm = []

    def lstm_hook(self, input, output):
        res = compute_lstm_flops(self, input, output)
        list_lstm.append(res)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.ReLU6):
                net.register_forward_hook(relu6_hook)
            if isinstance(net, torch.nn.MaxPool2d):
                net.register_forward_hook(max_pooling_hook)
            if isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(avg_pooling_hook)
            if isinstance(net, torch.nn.Softmax):
                net.register_forward_hook(softmax_pooling_hook)
            if isinstance(net, torch.nn.Sigmoid):
                net.register_forward_hook(sigmoid_pooling_hook)
            if isinstance(net, torch.nn.LSTM):
                net.register_forward_hook(lstm_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = torch.rand(batch_size, 3, input_res, input_res)
    if cuda:
        input = input.cuda()
    with torch.no_grad():
        out = model(input)
    total_flops = sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling)
    additional_model_flops = model.get_additional_flops()
    if 1 < len(out):
        prediction_flops = out[0].nelement() - 1
    else:
        prediction_flops = out.nelement() - 1
    additional_flops = additional_model_flops + prediction_flops
    total_flops += additional_flops

    def _rm_hooks(model):
        for m in model.modules():
            m._forward_hooks = OrderedDict()

    _rm_hooks(model)
    torch.cuda.empty_cache()
    gc.collect()
    return total_flops/batch_size

