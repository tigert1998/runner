from datetime import timedelta
import os
import ctypes
import random
import logging
import os.path as osp
from glob import glob
import shutil
from typing import Callable, List, Optional, Tuple, Dict, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .utils import config_logger, config_logger_handler, get_dist_info, set_seed, is_distributed
from .hooks import Hook
from .log_buffer import LogBuffer

# (rank, world_size, device_id, num_epochs, vars)
# (model, optimizer, lr_scheduler, train_data_loader, test_data_loaders)
DataLoaders = Union[DataLoader, Dict[str, DataLoader]]
DistTrainBuildFunction = Callable[
    [int, int, int, int, dict],
    Tuple[nn.Module, Optimizer, _LRScheduler, DataLoader, DataLoaders]
]


def _single_process_train(
    rank,
    world_size,
    work_dir,
    num_epochs,
    get_device_id_fn: Callable[[int, int, dict], int],
    dist_train_build_fn: DistTrainBuildFunction,
    port,
    seed: Optional[int],
    vars: Optional[dict],
    hooks: Optional[list],
):
    if seed is not None:
        set_seed(seed)
    if vars is None:
        vars = {}
    if hooks is None:
        hooks = []

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend='gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank,
        timeout=timedelta(days=1)
    )
    device_id = get_device_id_fn(rank, world_size, vars)
    torch.cuda.set_device(device_id)
    model, optimizer, lr_scheduler, train_data_loader, test_data_loaders = \
        dist_train_build_fn(
            rank, world_size, device_id,
            num_epochs, vars
        )
    ddp_model = DistributedDataParallel(
        model, device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=vars.get("find_unused_parameters", False)
    )
    train(
        ddp_model, work_dir, optimizer, lr_scheduler,
        num_epochs, train_data_loader, test_data_loaders, hooks,
        cuda_limit_malloc_heap_size=vars.get(
            "cuda_limit_malloc_heap_size", None)
    )


def dist_train(
    world_size,
    work_dir,
    num_epochs,
    get_device_id_fn: Callable[[int, int, dict], int],
    dist_train_build_fn: DistTrainBuildFunction,
    port: Optional[int] = None,
    seed: Optional[int] = None,
    vars: Optional[dict] = None,
    hooks: Optional[list] = None,
):
    if port is None:
        port = random.randint(10000, 20000)
    mp.spawn(_single_process_train, args=(
        world_size,
        work_dir,
        num_epochs,
        get_device_id_fn,
        dist_train_build_fn,
        port,
        seed,
        vars,
        hooks,
    ), nprocs=world_size)


class Runner:
    work_dir: str
    inner_iter: int
    iter: int
    num_iters: int
    epoch: int
    num_epochs: int
    flow: str
    model: nn.Module
    optimizer: Optimizer
    lr_scheduler: _LRScheduler
    outputs: dict
    log_buffer: LogBuffer
    hooks: List[Hook]
    logger: logging.Logger


def _run_iter(runner: Runner, train_mode, data_batch):
    model = runner.model.module if is_distributed() else runner.model
    method = model.train_step if train_mode else model.val_step
    runner.outputs = method(runner.model, data_batch, runner.optimizer)

    if train_mode and 'log_vars' in runner.outputs:
        runner.log_buffer.update(
            runner.outputs['log_vars'],
            runner.outputs['num_samples']
        )


def _call_hook(runner: Runner, fn):
    for hook in runner.hooks:
        getattr(hook, fn)(runner)


def _merge_outputs_across_processes(runner: Runner, outputs):
    rank, world_size = get_dist_info()
    if world_size == 1:
        return outputs
    dir_path = f"{runner.work_dir}/.outputs"
    if rank == 0:
        os.mkdir(dir_path)
    dist.barrier()
    torch.save(outputs, f"{dir_path}/{rank}.pth")
    dist.barrier()
    if rank > 0:
        return None
    outputs = ([], [])
    for i in range(world_size):
        tmp = torch.load(f"{dir_path}/{i}.pth", map_location="cpu")
        outputs[0].extend(tmp[0])
        outputs[1].extend(tmp[1])
    shutil.rmtree(dir_path)
    return outputs


def _evaluate_outputs(runner: Runner, outputs):
    rank, _ = get_dist_info()
    if rank > 0:
        return
    model = runner.model.module if is_distributed() else runner.model
    outputs = model.evaluate(*outputs)
    for var, val in outputs.items():
        runner.log_buffer.output[var] = val


def try_resume(runner: Runner):
    if not osp.isdir(runner.work_dir):
        return None, None
    paths = glob(f"{runner.work_dir}/epoch_*.pth")
    epochs = []
    for path in paths:
        try:
            epoch = int(path[path.rfind("_") + 1: path.rfind(".")])
            epochs.append(epoch)
        except:
            continue
    epochs = sorted(epochs, reverse=True)
    runner.logger.info(f"Found existing checkpoints: {epochs}")

    model = runner.model.module if is_distributed() else runner.model
    device = model.device

    for epoch in epochs:
        try:
            ckpt = f"{runner.work_dir}/epoch_{epoch}.pth"
            ckpt = torch.load(ckpt, map_location=device)
        except:
            continue
        return epoch, ckpt

    return None, None


def train(
    model, work_dir,
    optimizer,
    lr_scheduler,
    num_epochs, train_data_loader, test_data_loaders,
    hooks: List[Hook],
    cuda_limit_malloc_heap_size=None,
):
    if isinstance(test_data_loaders, DataLoader):
        test_data_loaders = {
            "default": test_data_loaders
        }

    if cuda_limit_malloc_heap_size is not None:
        assert 0 == ctypes.CDLL('libcudart.so').cudaDeviceSetLimit(
            ctypes.c_int(2),
            ctypes.c_size_t(cuda_limit_malloc_heap_size)
        )

    runner = Runner()
    runner.work_dir = work_dir
    runner.inner_iter = 0
    runner.iter = 0
    runner.num_iters = num_epochs * len(train_data_loader)
    runner.epoch = 0
    runner.num_epochs = num_epochs
    runner.flow = None
    runner.model = model
    runner.optimizer = optimizer
    runner.lr_scheduler = lr_scheduler
    runner.outputs = {}
    runner.log_buffer = LogBuffer()
    runner.hooks = hooks

    rank, _ = get_dist_info()

    runner.logger = config_logger("train", None)
    if rank > 0:
        runner.logger.handlers = []

    epoch, ckpt = try_resume(runner)
    resume = epoch is not None
    if resume:
        runner.iter = epoch * len(train_data_loader)
        runner.epoch = epoch
        model = runner.model.module if is_distributed() else runner.model
        model.load_state_dict(ckpt["state_dict"])
        runner.optimizer.load_state_dict(ckpt["optimizer"])
        runner.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    if rank == 0:
        if not resume:
            os.makedirs(work_dir, exist_ok=True)
        handler = logging.FileHandler(osp.join(work_dir, "log.txt"))
        config_logger_handler(handler)
        runner.logger.handlers.append(handler)

    runner.logger.info(f"Model Summary:\n{model}")

    _call_hook(runner, "before_run")

    while runner.epoch < runner.num_epochs:
        runner.flow = "train"
        runner.log_buffer.clear()
        model.train()
        if is_distributed():
            train_data_loader.sampler.set_epoch(runner.epoch)
        runner.inner_iter = 0
        _call_hook(runner, "before_train_epoch")
        for data_batch in train_data_loader:
            _call_hook(runner, "before_train_iter")
            _run_iter(runner, True, data_batch)
            _call_hook(runner, "after_train_iter")
            runner.lr_scheduler.step()
            runner.inner_iter += 1
            runner.iter += 1

        _call_hook(runner, "after_train_epoch")
        runner.epoch += 1

        model.eval()
        for name, test_data_loader in test_data_loaders.items():
            runner.flow = f"val/{name}"
            runner.log_buffer.clear()
            runner.inner_iter = 0
            outputs = ([], [])
            _call_hook(runner, "before_val_epoch")
            for data_batch in test_data_loader:
                _call_hook(runner, "before_val_iter")
                with torch.no_grad():
                    _run_iter(runner, False, data_batch)
                _call_hook(runner, "after_val_iter")
                outputs[0].append(runner.outputs[0])
                outputs[1].append(runner.outputs[1])
                runner.inner_iter += 1

            outputs = _merge_outputs_across_processes(runner, outputs)
            _evaluate_outputs(runner, outputs)

            _call_hook(runner, "after_val_epoch")

    _call_hook(runner, "after_run")
