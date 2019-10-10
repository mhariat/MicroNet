from utils.data_utils import *
from utils.compute_flops import *
from utils.quantization_utils import *
from models.pyramid_net import *
from models.pyramid_skipnet import *
import argparse


def init_network(config):
    if config.network == 'pyramidnet':
        net = PyramidNet(dataset='cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    elif config.network == 'pyramidskipnet':
        net = PyramidSkipNet(dataset='cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    else:
        raise NotImplementedError
    path_to_add = '/usr/share/bind_mount/scripts/MicroNet/Sparsity/sparse_weights/{}/{}'.format(config.network,
                                                                                                config.exp_name)
    checkpoint_file = config.checkpoint_file_sparsity
    assert os.path.exists(path_to_add), 'No pruning for the corresponding network and/or experience'
    if 'skip' in net.name.lower():
        if config.alpha == 1e-4:
            alpha = 4
        elif config.alpha == 1e-5:
            alpha = 5
        else:
            raise NotImplementedError
        checkpoint_path = '{}/alpha_{}/{}'.format(path_to_add, alpha, checkpoint_file)
        exp_name = config.exp_name
    else:
        checkpoint_path = '{}/{}'.format(path_to_add, checkpoint_file)
        exp_name = config.exp_name

    net = load_checkpoint_pruning(checkpoint_path, net, use_bias=True)
    if config.freebie:
        compression = checkpoint_file.split('_')[-2]
        sparsity = checkpoint_file.split('_')[-1].split('.pth')[0]
        return net, exp_name, float(compression), float(sparsity)
    else:
        a_quant_module = SigmaMax(forward_bits=12, forward_avg_const=0.01, forward_num_std=20)
        w_quant_module = SigmaMax(forward_bits=9, forward_avg_const=1.0, forward_num_std=10)
        quant_net = QuantizedNet(net, a_quant_module, w_quant_module)
        load_quantization_architecture(quant_net)
        path_to_add = '/usr/share/bind_mount/scripts/MicroNet/Quantization/quantized_weights/{}/{}'.\
            format(config.network, config.exp_name)
        checkpoint_file = config.checkpoint_file_quantization
        checkpoint_path = '{}/{}'.format(path_to_add, checkpoint_file)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        quant_net.load_state_dict(checkpoint)
        if torch.cuda.is_available():
            quant_net.cuda()
        compression = checkpoint_file.split('_')[-2]
        sparsity = checkpoint_file.split('_')[-1].split('.pth')[0]
        return quant_net, exp_name, float(compression), float(sparsity)


def main(config):

    print(config)
    train_dataloader, val_dataloader = get_dataloader(batch_size=config.batch_size, num_workers=config.num_workers)

    net, exp_name, compression, sparsity = init_network(config)
    loss_function = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accuracy = test(net, val_dataloader, loss_function, device).item()

    total_flop_mults, total_flop_adds, total_flop_exponent = get_total_flops(net, input_res=32,
                                                                             test_dataloader=val_dataloader)
    total_non_zero_parameters = sum(torch.gt(torch.abs(p.data), 0).sum().item() for p in net.parameters())/1e6
    total_parameters = count_all_parameters(net)/1e6

    if config.freebie:
        mul_bits = 16
        add_bits = 32
        param_bits = 16
        exponent_bits = 0
    else:
        mul_bits = config.mul_bits
        add_bits = config.add_bits
        param_bits = config.param_bits
        exponent_bits = config.exponent_bits

    storage = total_non_zero_parameters*param_bits/32 + total_parameters*1/32
    math_ops = total_flop_adds*add_bits/32 + total_flop_mults*mul_bits/32 + total_flop_exponent*exponent_bits/32
    micronet_score = storage/36.5 + math_ops/10.49
    print("Final accuracy: ", accuracy)
    print("Micronet Score: ", micronet_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the config json file')
    args = parser.parse_args()
    config_path = args.config_path
    config = get_config_from_json(config_path)
    main(config)
