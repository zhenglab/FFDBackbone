import numpy as np
import math

import omegaconf


def to_list(inp):
    if inp is None:
        return None
    if isinstance(inp, omegaconf.listconfig.ListConfig):
        inp = omegaconf.OmegaConf.to_container(inp, resolve=True)
    return inp


def lr_tuner(lr_init, optimizer, epoch_size, tune_dict, global_step=1, use_warmup=False, warmup_epochs=1, lr_min=1e-6):
    """A simple learning rate tuning strategy.
    Using tune_dict to tune the learning rate.
    e.g.:
        tune_dict: 
            key: strategy name, value: strategy params
            (1) piecewise:
            {
                "decay_steps": [100, 200],
                "decay_epochs": [1, 2],
                "decay_rates": [0.1, 0.2],
            }
            (2) exponential: 
            {
                "decay_step": 100,
                "decay_epoch": 1,
                "decay_rate": 0.9,
                "staircase": True,
            }

    Args:
        lr_init (float): Initial learning rate.
        optimizer (torch.optimizer): Torch optimizer.
        epoch_size (int): How many steps in one epoch.
        tune_dict (dict): The dict specifiying the learning rate tuning strategy.
        global_step (int, optional): Global step. Defaults to 1.
        use_warmup (bool, optional): Setting True to use warmup strategy. Defaults to False.
        warmup_epochs (int, optional): How many epoches to apply warmup. Defaults to 1.
        lr_min (float, optional): Minimal learning rate. Defaults to 1e-6.

    Returns:
        float: The tuned learning rate.
    """
    if use_warmup and global_step <= epoch_size * warmup_epochs:
        lr_start = 0
        if global_step == 1:
            print(">>> Using warmup strategy!")
        # lr = global_step / epoch_size * warmup_epochs * lr_init
        alpha = (lr_init - lr_start) / (epoch_size * warmup_epochs)
        lr = global_step * alpha + lr_start
    else:
        if use_warmup:
            new_step = global_step - epoch_size * warmup_epochs
        else:
            new_step = global_step

        tune_dict = dict(tune_dict)
        decay_strategy_name = tune_dict['name']

        if decay_strategy_name == "piecewise":
            decay_steps = to_list(tune_dict.get("decay_steps"))
            decay_epochs = to_list(tune_dict.get("decay_epochs"))
            decay_rates = to_list(tune_dict.get("decay_rates"))

            if decay_steps and decay_epochs:
                raise ValueError(
                "decay_steps and decay_epochs in tune_dict of lr_tuner are both set, only one of them can be set"
                )
            if decay_epochs:
                decay_steps = [epoch_size * x for x in decay_epochs]
            
            decay_cnt = np.sum(new_step > np.asarray(decay_steps))
            if decay_cnt == 0:
                decay_mult = 1.0
            else:
                decay_mult = decay_rates[decay_cnt - 1]
            
            lr = max(lr_init * decay_mult, lr_min)

        elif decay_strategy_name == "exponential":
            decay_step = tune_dict.get("decay_step")
            decay_epoch = tune_dict.get("decay_epoch")
            decay_rate = tune_dict.get("decay_rate")
            staircase = tune_dict.get("staircase")

            if decay_step and decay_epoch:
                raise ValueError(
                "decay_step and decay_epoch in tune_dict of lr_tuner are both set, only one of them can be set"
                )
            if decay_epoch:
                decay_step = epoch_size * decay_epoch
            
            decay_index = new_step // decay_step if staircase else new_step / decay_step
            lr = max(lr_init * math.pow(decay_rate, decay_index), lr_min)
        elif decay_strategy_name == "linear":
            decay_epoch = tune_dict.get("decay_epochs")
            epochs = tune_dict.get("epochs")
            if decay_epoch:
                decay_step = epoch_size * decay_epoch
            alpha = 1.0 - max(0, global_step - decay_step) / float(epoch_size*epochs - decay_step)
            lr = lr_init*alpha
        else:
            raise NotImplementedError("decay_strategy_name {} is not supported".format(decay_strategy_name))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

from torch.optim.lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, n_epoch, start_decay, last_epoch=-1):
        self.start_decay=start_decay
        self.n_epoch=n_epoch
        super(LinearDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        last_epoch = self.last_epoch
        n_epoch=self.n_epoch
        b_lr=self.base_lrs[0]
        start_decay=self.start_decay
        if last_epoch>start_decay:
            lr=b_lr-b_lr/(n_epoch-start_decay)*(last_epoch-start_decay)
        else:
            lr=b_lr
        return [lr]

