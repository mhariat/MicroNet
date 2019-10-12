from utils.data_utils import *
from models.pyramid_net import *
from models.pyramid_skipnet import *
from nn.quantization.utils.utils import calibrate
import nn.quantization as quant
import utils.utils as utils
import torch.nn as nn
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
    checkpoint_file = config.checkpoint_file
    assert os.path.exists(path_to_add), 'No pruning for the corresponding network and/or experience'
    checkpoint_path = '{}/{}'.format(path_to_add, checkpoint_file)
    exp_name = config.exp_name
    net = load_checkpoint_pruning(checkpoint_path, net, use_bias=True)
    utils.remove_bn(net)
    if torch.cuda.is_available():
        net.cuda()
    compression = checkpoint_file.split('_')[-2]
    sparsity = checkpoint_file.split('_')[-1].split('.pth')[0]
    return net, exp_name, float(compression), float(sparsity)


def main(config):

    print(config)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device_used = 'GPU'
    else:
        device_used = 'CPU'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader = get_dataloader(batch_size=config.batch_size, num_workers=config.num_workers)

    num_classes = len(train_dataloader.dataset.classes)
    net, exp_name, compression, sparsity = init_network(config)

    # Define loss function
    loss_function = nn.CrossEntropyLoss()
    float_accuracy = utils.test(net, val_dataloader, loss_function, device)

    dataset = 'cifar_100'
    path_to_add = '{}/{}/{}/{}'.format(dataset, config.network, 272., exp_name)
    exp_name = config.exp_name
    checkpoint_dir = '{}/checkpoint/quantization/{}'.format(config.result_dir, path_to_add)
    create_dir(checkpoint_dir)

    message = '-'*200
    print(message)
    print(message)

    message = 'Quantization part. Device used: {}. Dataset: {}. Number of classes: {}. Network to prune: {}_{}.' \
              ' Pruned Network accuracy: {:.2f}%.' \
              ' (Exp_name: {}. Network Compression: {:.2f}%. Network Sparsity: {:.2f}%)'.\
        format(device_used, dataset, num_classes, config.network, 272, 100*float_accuracy, exp_name,
               100*compression, 100*sparsity)
    print(colored(message, 'magenta'))

    message = '-' * 200
    print(message)
    print(message)

    activation_bits = config.activation_bits
    activation_num_std = config.activation_num_std
    weight_bits = config.weight_bits
    weight_num_std = config.weight_num_std
    # Quantize network
    a_quant_module = quant.SigmaMax(forward_bits=activation_bits, forward_avg_const=0.01,
                                    forward_num_std=activation_num_std)
    w_quant_module = quant.SigmaMax(forward_bits=weight_bits, forward_avg_const=1.0, forward_num_std=weight_num_std)

    quant_net = quant.QuantizedNet(net, a_quant_module, w_quant_module)

    # Calibrate quantized network
    calibrate(quant_net, val_dataloader, loss_function, device)

    quant_accuracy = utils.test(quant_net, val_dataloader, loss_function, device)
    print('Quantized network accuracy %.3f ' % (quant_accuracy * 100))

    filename = 'checkpoint_quantized_weights_{:.4f}_{:.4f}_{:.4f}.pth'.format(quant_accuracy, compression, sparsity)
    torch.save(quant_net.state_dict(), '{}/{}'.format(checkpoint_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the config json file')
    args = parser.parse_args()
    config_path = args.config_path
    config = get_config_from_json(config_path)
    main(config)