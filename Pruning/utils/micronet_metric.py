from utils.compute_flops import *
from utils.common_utils import *


wideresnet_cifar100 = {'storage': 36.5, 'flops': 10490}
mobilenetv2_imagenet = {'storage': 6.108776, 'flops': 1191.778551}


def score(model, input_res, num_classes, batch_size):
    if num_classes == 100:
        bench_flops = wideresnet_cifar100['flops']
        bench_storage = wideresnet_cifar100['storage']
    elif num_classes == 1000:
        bench_flops = mobilenetv2_imagenet['flops']
        bench_storage = mobilenetv2_imagenet['storage']
    else:
        raise NotImplementedError
    flops = get_total_flops(model, input_res, batch_size=batch_size)/1e6
    storage = (count_all_parameters(model) + model.get_additional_parameters())/1e6
    return flops/bench_flops + storage/bench_storage

