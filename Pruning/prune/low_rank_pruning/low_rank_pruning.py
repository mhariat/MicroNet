from prune.pruning import *


class LowRankPruner(Pruner, metaclass=ABCMeta):
    def __init__(self, model, prune_ratio_limit, normalize, log_interval, use_hook):
        super(LowRankPruner, self).__init__(model=model, prune_ratio_limit=prune_ratio_limit, normalize=normalize,
                                            log_interval=log_interval, use_hook=use_hook)
        self.module_with_dependencies = model.get_module_with_dependencies()

    def get_nb_parameters(self):
        return sum(p.numel() for p in self.model.parameters())
