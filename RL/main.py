from models.pyramid_skipnet import *
from utils.data_utils import *
from utils.log import *
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import argparse


def init_network(config):
    net = PyramidSkipNet(dataset='cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    path_to_add = '/usr/share/bind_mount/scripts/MicroNet/Pruning/pruned_weights/{}/{}'.format(config.network,
                                                                                               config.exp_name)
    checkpoint_file = config.checkpoint_file
    assert os.path.exists(path_to_add), 'No pruning for the corresponding network and/or experience'
    checkpoint_path = '{}/{}'.format(path_to_add, checkpoint_file)
    exp_name = config.exp_name
    net = load_checkpoint_pruning(checkpoint_path, net, use_bias=True)
    if torch.cuda.is_available():
        net.cuda()
    compression = checkpoint_file.split('_')[-1].split('.pth')[0]
    return net, exp_name, float(compression)


def main(config):
    print(config)

    train_dataloader, val_dataloader = get_dataloader(batch_size=config.batch_size, num_workers=config.num_workers)
    num_classes = len(train_dataloader.dataset.classes)
    net, exp_name, compression = init_network(config)

    val_acc, _, skip_ratios = validation_rl(model=net, use_cuda=torch.cuda.is_available(),
                                            val_dataloader=val_dataloader)
    original_accuracy = val_acc
    skip_summaries = []
    for idx in range(skip_ratios.len):
        skip_summaries.append(1 - skip_ratios.avg[idx])
    cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
    original_cp = cp.item()

    create_dir(config.result_dir)
    dataset = 'cifar_100'
    path_to_add = '{}/{}/{}'.format(dataset, config.network, 272)
    writer_dir = '{}/writer/RL/{}'.format(config.result_dir, path_to_add)
    exp_name = config.exp_name
    if exp_name is None:
        exp_name = 'unknown'
    run = get_run(writer_dir, exp_name)
    exp_name = 'run_{}'.format(run)
    writer_dir = '{}/{}'.format(writer_dir, exp_name)
    checkpoint_dir = '{}/checkpoint/RL/{}'.format(config.result_dir, path_to_add)
    log_dir = '{}/logs/RL/{}'.format(config.result_dir, path_to_add)
    create_dir(writer_dir)
    create_dir(checkpoint_dir)
    writer = SummaryWriter(writer_dir)
    logger = get_logger('log_{}'.format(exp_name), log_dir)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device_used = 'GPU'
    else:
        device_used = 'CPU'
    lr, momentum, weight_decay, epochs = config.lr, config.momentum, config.weight_decay, config.epochs
    message = '-'*200
    print(message)
    print(message)
    message = 'RL part. Device used: {}. Dataset: {}. Number of classes: {}. Network to prune: {}_{}.' \
              ' Pruned Network accuracy: {:.2f}%. Pruned Network Computation Percentage: {:.2f}' \
              ' (Exp_name: {}. Network Compression: {:.2f}%)'.\
        format(device_used, dataset, num_classes, config.network, 272, 100*original_accuracy, original_cp,
               exp_name, 100*compression)
    print(colored(message, 'magenta'))
    logger.info(message)
    logger.info('-'*150)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    starting_lr = optimizer.param_groups[0]['lr']
    lr = starting_lr
    total_rewards = AverageMeter()

    for epoch in range(epochs):
        message = '-' * 200
        print(message)
        print(message)
        train_acc, train_loss = train_rl(net, optimizer, use_cuda, train_dataloader, epoch, total_rewards, config.alpha,
                                         config.gamma, config.rl_weight, config.log_interval)
        val_acc, val_loss, skip_ratios = validation_rl(model=net, use_cuda=use_cuda, val_dataloader=val_dataloader)
        skip_summaries = []
        for idx in range(skip_ratios.len):
            skip_summaries.append(1 - skip_ratios.avg[idx])
        cp = ((sum(skip_summaries) + 1) / (len(skip_summaries) + 1)) * 100
        message = 'Training. Epoch: [{}/{}]. Train Loss: {:.6f}. Train Accuracy: {:.2f}%. Learning rate {:.5f}.' \
                  ' Validation loss: {:.6f}. Validation Accuracy: {}/{} ({:.2f}%). Computation Percentage: {:.3f} %'.\
            format(epoch, epochs, train_loss, 100 * train_acc, lr, val_loss, int(100*val_acc),
                   len(val_dataloader.dataset), 100. * val_acc, cp.item())
        print(colored('\n{}\n'.format(message), 'yellow'))
        logger.info(message)
        stat = {
            'Performance/train_loss': train_loss,
            'Performance/train_acc': train_acc,
            'Performance/val_loss': val_loss,
            'Performance/val_acc': val_acc,
        }
        for name in stat.keys():
            writer.add_scalar(name, stat[name], epoch)

        if epoch % 11 == 0:
            filename = 'checkpoint_{}_{:.4f}_{:.4f}_{:.4f}.pth'.format(exp_name, val_acc, compression, cp.item())
            torch.save(net.state_dict(), '{}/{}'.format(checkpoint_dir, filename))

    filename = 'checkpoint_{}_{:.4f}_{:.4f}_{:.4f}.pth'.format(exp_name, val_acc, compression, cp.item())
    torch.save(net.state_dict(), '{}/{}'.format(checkpoint_dir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the config json file')
    args = parser.parse_args()
    config_path = args.config_path
    config = get_config_from_json(config_path)
    main(config)
