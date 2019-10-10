from models.pyramid_net import *
from models.pyramid_skipnet import *
from prune.weights_pruning.methods.magnitude import *
from utils.data_utils import *
from utils.results_utils import *
from torch import optim
import argparse

scheduler_id = {
    0: 'Exponential',
    1: 'Stair',
    2: 'UpDown'
}


def init_network(config):
    if config.network == 'pyramidnet':
        net = PyramidNet(dataset='cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    elif config.network == 'pyramidskipnet':
        net = PyramidSkipNet(dataset='cifar100', depth=272, alpha=200, num_classes=100, bottleneck=True)
    else:
        raise NotImplementedError
    path_to_add = '/usr/share/bind_mount/scripts/MicroNet/Pruning/pruned_weights/{}/{}'.format(config.network,
                                                                                               config.exp_name)
    checkpoint_file = config.checkpoint_file
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
    if torch.cuda.is_available():
        net.cuda()
    compression = checkpoint_file.split('_')[-1].split('.pth')[0]
    return net, exp_name, float(compression)


def init_pruner(config, network):
    kwargs = {'model': network, 'prune_ratio_limit': config.prune_ratio_limit, 'log_interval': config.log_interval}
    pruner = MagnitudePruner(**kwargs)
    dataset = 'cifar_100'
    pruner.dataset = dataset
    pruner.name = "Magnitude"
    return pruner


def init_scheduler(config):
    if config.scheduler_id == 0:
        return ExpLRScheduler()
    elif config.scheduler_id == 1:
        return StairLRScheduler()
    elif config.scheduler_id == 2:
        return UpDownLRScheduler()


def main(config):
    print(config)
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        device_used = 'GPU'
    else:
        device_used = 'CPU'
    train_dataloader, val_dataloader = get_dataloader(batch_size=config.batch_size, num_workers=config.num_workers)

    num_classes = len(train_dataloader.dataset.classes)
    net, exp_name, compression = init_network(config)
    scheduler = init_scheduler(config)
    pruner = init_pruner(config, net)
    pruner_name = pruner.name
    pruner.compression = compression
    if 'exp_name' in config:
        exp_name_results = config.exp_name
    else:
        exp_name_results = None
    result = SparsityResults(config.result_dir, train_dataloader, val_dataloader, pruner, exp_name_results)
    logger = result.logger
    scheduler_params = {float(x): y for x, y in config.scheduler_params.items()}
    lr, momentum, weight_decay = config.lr, config.momentum, config.weight_decay
    normalize, prune_ratio, prune_ratio_limit = True, config.prune_ratio, config.prune_ratio_limit
    epochs_start, epochs_end = config.epochs_start, config.epochs_end
    original_accuracy = result.stats[0]['Performance/val_acc']
    pruner.saved_model.append((original_accuracy, pruner.original_model))
    dataset = 'cifar_100'
    message = 'Sparsity method used: {}. Scheduler used: {}. Device used: {}. Dataset: {}. Number of classes: {}.' \
              ' Network to prune: {}_{}. Pruned Network accuracy: {:.2f}% (Exp_name: {}.' \
              ' Network Compression: {:.2f}%)'.\
        format(pruner_name, scheduler_id[config.scheduler_id], device_used, dataset, num_classes, config.network,
               config.depth, 100*original_accuracy, exp_name, 100*compression)
    print(colored('\n{}\n'.format(message), 'magenta'))
    logger.info(message)
    it = 0
    max_it = config.max_iter
    stop_pruning = False

    while not stop_pruning:
        message = '-' * 200
        print(message)
        print(message)
        logger.info('-' * 150)
        message = '[Sparsity method: {}. Iteration: {}]. Pruning. Prune-ratio: {}. Prune-ratio limit: {}'.\
            format(pruner_name, it, prune_ratio, prune_ratio_limit)
        print(colored(message, 'magenta'))
        logger.info(message)
        pruner.prune(prune_ratio=prune_ratio, train_dataloader=train_dataloader)
        initial_nb_parameters = result.stats[0]['Other/total_parameters']
        current_compression = (initial_nb_parameters - pruner.get_nb_parameters())/initial_nb_parameters
        epochs = int(ratio_interpolation(start_value=epochs_start, end_value=epochs_end, ratio=current_compression))
        lr_schedule = {int(x*epochs): y for x, y in scheduler_params.items()}
        lr = config.lr
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler.settle(lr_schedule=lr_schedule, optimizer=optimizer)
        pruner.fine_tune(epochs=epochs, optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader, alpha=config.alpha)
        result.add_results()
        current_compression = result.stats[it + 1]['Ratio/prune_ratio']
        current_accuracy = result.stats[it + 1]['Performance/val_acc']
        message = '[Sparsity method: {}. Iteration: {}]. Accuracy: {:.2f}% [Original Accuracy: {:.2f}%].' \
                  ' Cumulative Compression: {:.2f}%'.format(pruner_name, it, 100*current_accuracy,
                                                            100*original_accuracy, 100*current_compression)
        if pruner.skip:
            cp = result.stats[it + 1]['Indicator/Computation_percentage']
            message = '{}. Cumulative percentage: {:.3f}.'.format(message, cp)
        print(colored(message, 'green'))
        logger.info(message)
        torch.cuda.empty_cache()
        gc.collect()
        it += 1
        stop_pruning = (max_it <= it) | (current_accuracy < 0.8)
    result.clean_up()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to the config json file')
    args = parser.parse_args()
    config_path = args.config_path
    config = get_config_from_json(config_path)
    main(config)
