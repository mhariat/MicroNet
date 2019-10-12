from models.pyramid_net import *
from models.pyramid_skipnet import *
from utils.results_utils import *
from prune.low_rank_pruning.methods.improved_eigen_damage import *
from prune.low_rank_pruning.methods.improved_eigen_damage_multi import *
from utils.data_utils import *
from torch import optim
import argparse


pruner_id = {
    0: 'Improved-Eigen-Damage',
    1: 'Improved-Eigen-Damage_multi',
}

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
    path_to_add = '/usr/share/bind_mount/scripts/MicroNet/Training/trained_weights/{}'.format(config.network)
    if 'skip' in net.name.lower():
        alpha = 4
        checkpoint_path = '{}/alpha_{}/pyramid_skip_last_epoch_alpha_{}.pth'.format(path_to_add, alpha, alpha)
        exp_name = 'PyramidSkipNet'
    else:
        checkpoint_path = '{}/pyramid_last_epoch.pth'.format(path_to_add)
        exp_name = 'PyramidNet'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['model']
    net.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        net.cuda()
    return net, exp_name, 1800


def init_pruner(config, network):
    kwargs = {'model': network, 'prune_ratio_limit': config.prune_ratio_limit, 'log_interval': config.log_interval}
    if config.pruner_id in [0, 1]:
        if 'back_ratio' in config:
            kwargs.update({'back_ratio': config.back_ratio})
        if config.pruner_id == 0:
            pruner = ImprovedEigenPruner(**kwargs)
        else:
            pruner = ImprovedEigenPrunerMulti(**kwargs)
    else:
        raise NotImplementedError
    dataset = 'cifar_100'
    pruner.dataset = dataset
    pruner.name = pruner_id[config.pruner_id]
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
    train_dataloader_pruning, train_dataloader, val_dataloader = get_dataloader(batch_size=128,
                                                                                batch_size_pruning=100,
                                                                                num_workers=config.num_workers)

    num_classes = len(train_dataloader.dataset.classes)
    net, exp_name, original_epochs = init_network(config)
    scheduler = init_scheduler(config)
    pruner = init_pruner(config, net)
    if 'exp_name' in config:
        exp_name_results = config.exp_name
    else:
        exp_name_results = None
    result = PruneResults(config.result_dir, train_dataloader, val_dataloader, pruner, exp_name_results)
    logger = result.logger
    scheduler_params = {float(x): y for x, y in config.scheduler_params.items()}
    lr, momentum, weight_decay = config.lr, config.momentum, config.weight_decay
    normalize, prune_ratio, prune_ratio_limit = config.normalize, config.prune_ratio, config.prune_ratio_limit
    epochs_start, epochs_end = config.epochs_start, config.epochs_end
    original_accuracy = result.stats[0]['Performance/val_acc']
    pruner.saved_model.append((original_accuracy, pruner.original_model))
    dataset = 'cifar_100'
    message = 'Pruning method used: {}. Scheduler used: {}. Device used: {}. Dataset: {}. Number of classes: {}.' \
              ' Network to prune: {}_{}. Original accuracy: {:.2f}% (Exp_name: {}. Trained for {} epochs)'.\
        format(pruner_id[config.pruner_id], scheduler_id[config.scheduler_id], device_used, dataset, num_classes,
               config.network, 272, 100*original_accuracy, exp_name, original_epochs)
    print(colored('\n{}\n'.format(message), 'magenta'))
    logger.info(message)
    it = 0
    max_it = config.max_iter
    stop_pruning = False

    trigger_back = False
    trigger_compression_1 = False
    trigger_compression_2 = False
    trigger_compression_3 = False

    batches_no_skip = [(168, 128), (176, 128), (176, 128)]
    batches_skip = [(144, 128), (152, 128), (160, 128)]

    if pruner.skip:
        batches = batches_skip
    else:
        batches = batches_no_skip

    while not stop_pruning:
        message = '-' * 200
        print(message)
        print(message)
        logger.info('-' * 150)
        message = '[Pruning method: {}. Iteration: {}]. Pruning. Prune-ratio: {}. Prune-ratio limit: {}'.\
            format(pruner_id[config.pruner_id], it, prune_ratio, prune_ratio_limit)
        print(colored(message, 'magenta'))
        logger.info(message)
        if hasattr(pruner, 'pool'):
            pool_it = pruner.pool.it
        else:
            pool_it = 0
        pruner.prune(prune_ratio=prune_ratio, train_dataloader=train_dataloader_pruning)
        initial_nb_parameters = result.stats[0]['Other/total_parameters']
        current_compression = (initial_nb_parameters - pruner.get_nb_parameters())/initial_nb_parameters

        if 0 < current_compression:
            if not trigger_compression_1:
                train_dataloader_pruning, train_dataloader, val_dataloader = \
                    get_dataloader(batch_size=batches[0][0], batch_size_pruning=batches[0][1],
                                   num_workers=config.num_workers)
            trigger_compression_1 = True
            trigger_back = True

        if 0.3 < current_compression:
            if not trigger_compression_2:
                train_dataloader_pruning, train_dataloader, val_dataloader = \
                    get_dataloader(batch_size=batches[1][0], batch_size_pruning=batches[1][1],
                                   num_workers=config.num_workers)
            trigger_compression_2 = True

        if 0.6 < current_compression:
            if not trigger_compression_3:
                train_dataloader_pruning, train_dataloader, val_dataloader = \
                    get_dataloader(batch_size=batches[2][0], batch_size_pruning=batches[2][1],
                                   num_workers=config.num_workers)
            trigger_compression_3 = True

        if 0 < current_compression:
            epochs = int(ratio_interpolation(start_value=epochs_start, end_value=epochs_end,
                                             ratio=current_compression))
        else:
            epochs = config.negative_compression_epochs

        if not trigger_back:
            pruner.pool.it = 0

        if pool_it % 2 == 1:
            lr_schedule = {50: 0.1, 100: 0.01, 150: 0.001}
            lr = 10*config.lr
        else:
            lr_schedule = {int(x*epochs): y for x, y in scheduler_params.items()}
            lr = config.lr
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler.settle(lr_schedule=lr_schedule, optimizer=optimizer)
        pruner.fine_tune(epochs=epochs, optimizer=optimizer, scheduler=scheduler, train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader, alpha=config.alpha)
        result.add_results()
        current_compression = result.stats[it + 1]['Ratio/prune_ratio']
        current_accuracy = result.stats[it + 1]['Performance/val_acc']
        message = '[Pruning method: {}. Iteration: {}]. Accuracy: {:.2f}% [Original Accuracy: {:.2f}%].' \
                  ' Cumulative Compression: {:.2f}%'.format(pruner_id[config.pruner_id], it, 100*current_accuracy,
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
