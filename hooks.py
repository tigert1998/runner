import os
import os.path as osp

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import is_distributed, master_only


class Hook:
    def before_run(self, runner):
        ...

    def before_train_epoch(self, runner):
        ...

    def before_train_iter(self, runner):
        ...

    def after_train_iter(self, runner):
        ...

    def after_train_epoch(self, runner):
        ...

    def before_val_epoch(self, runner):
        ...

    def before_val_iter(self, runner):
        ...

    def after_val_iter(self, runner):
        ...

    def after_val_epoch(self, runner):
        ...

    def after_run(self, runner):
        ...

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False


class LoggerHook(Hook):
    def __init__(self, interval):
        self.reset_flag = False
        self.interval = interval

    def before_run(self, runner):
        for hook in runner.hooks[::-1]:
            if isinstance(hook, LoggerHook):
                hook.reset_flag = True
                break

    def get_loggable_tags(self, runner):
        tags = {}
        for var, val in runner.log_buffer.output.items():
            tags[f"{runner.flow}/{var}"] = val
        tags["learning_rate"] = runner.optimizer.param_groups[0]["lr"]
        return tags

    def log(self, runner):
        ...

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            runner.log_buffer.average(self.interval)
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_train_epoch(self, runner):
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

    def after_val_epoch(self, runner):
        runner.log_buffer.average()
        self.log(runner)
        if self.reset_flag:
            runner.log_buffer.clear_output()


class TensorboardLoggerHook(LoggerHook):
    def __init__(self, interval):
        super().__init__(interval)

    @master_only
    def before_run(self, runner):
        super().before_run(runner)
        self.writer = SummaryWriter(
            log_dir=osp.join(runner.work_dir, "tf_logs"))

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, runner.iter + 1)
            else:
                self.writer.add_scalar(tag, val, runner.iter + 1)

    @master_only
    def after_run(self, runner):
        self.writer.close()


class CheckpointHook(Hook):
    @master_only
    def after_train_epoch(self, runner):
        path = osp.join(runner.work_dir, f"epoch_{runner.epoch + 1}.pth")
        torch.save(
            runner.model.module.state_dict() if is_distributed() else runner.model.state_dict(),
            path
        )
        if runner.epoch - 4 >= 1:
            os.remove(osp.join(runner.work_dir,
                      f"epoch_{runner.epoch - 4}.pth"))
