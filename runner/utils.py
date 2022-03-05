import logging
import sys
import random
from typing import Optional
import functools

import numpy as np

import torch
import torch.distributed as dist


def config_logger_handler(handler):
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(filename)s:%(lineno)s] [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)


def config_logger(name, filename: Optional[str]):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if filename is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(filename)
    config_logger_handler(handler)
    logger.handlers = [handler]
    return logger


def is_distributed():
    return dist.is_initialized() and dist.is_available()


def get_dist_info():
    if is_distributed():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)
    return wrapper


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
