
import argparse
import logging
import torch
import torchvision
import numpy as np
import datetime
import time
from functools import wraps
from typing import Any, Callable
import os


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class LRScheduler(object):

    def __init__(self, optimizer, warm_up):
        super(LRScheduler, self).__init__()
        self.optimizer = optimizer
        self.warm_up = warm_up

    def update(self, iteration, learning_rate, num_group=1000, ratio=1):
        if iteration < self.warm_up:
            learning_rate *= iteration / self.warm_up
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = learning_rate * ratio**i


def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Times a function, usually used as decorator"""
    # ref: http://zyxue.github.io/2017/09/21/python-timeit-decorator.html
    @wraps(func)
    def timed_func(*args: Any, **kwargs: Any) -> Any:
        """Returns the timed function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
        print("time spent on %s: %s"%(func.__name__, elapsed_time))
        return result

    return timed_func


def to_grid_tex(x):
    with torch.no_grad():
        return torchvision.utils.make_grid((x.data[:, :3]+1)/2, nrow=4)

def to_grid_mesh(x):
    with torch.no_grad():
        x = x.data[:, :3]
        minv = x.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        maxv = x.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        x = (x - minv)/(maxv-minv)
        return torchvision.utils.make_grid(x, nrow=4)


def set_requires_grad(nets, requires_grad=False):
        # ref: https://github.com/lyndonzheng/F-LSeSim/blob/e092e62ed8a2f51f3661630e1522ec2549ec31d3/models/base_model.py#L229 
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad