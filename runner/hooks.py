import os
import os.path as osp
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter

from .utils import is_distributed, master_only


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
    def __init__(self, save_best=None, compare_op="greater"):
        super().__init__()
        self.save_best = save_best
        self.compare_op = compare_op
        self.best_score = {}
        self.best_ckpt_path = {}

    def compare(self, x, y):
        if self.compare_op == "greater":
            return x > y
        elif self.compare_op == "less":
            return x < y

    @staticmethod
    def _save_checkpoint(work_dir, state_dict, optimizer, lr_scheduler, epoch, meta, template="epoch_{}.pth"):
        path = osp.join(work_dir, template.format(epoch))
        torch.save({
            "meta": {"epoch": epoch, **meta},
            "state_dict": state_dict,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }, path)

    @master_only
    def after_train_epoch(self, runner):
        model = runner.model.module if is_distributed() else runner.model
        self._save_checkpoint(
            runner.work_dir,
            model.state_dict(),
            runner.optimizer,
            runner.lr_scheduler,
            runner.epoch + 1, {}
        )
        if runner.epoch - 4 >= 1:
            os.remove(osp.join(runner.work_dir,
                      f"epoch_{runner.epoch - 4}.pth"))

    @master_only
    def after_val_epoch(self, runner):
        if self.save_best is None:
            return

        flow = runner.flow.split('/')[-1]
        model = runner.model.module if is_distributed() else runner.model

        if self.best_score.get(flow, None) is None:
            # try to find the checkpoint with the best score
            paths = glob(
                f"{runner.work_dir}/{flow}_best_{self.save_best}_epoch_*.pth"
            )
            for path in paths:
                try:
                    ckpt = torch.load(path, map_location="cpu")
                except:
                    continue
                if ckpt.get("meta", None) is not None and \
                        ckpt["meta"].get(flow, None) is not None and \
                        ckpt["meta"][flow].get(self.save_best, None) is not None and \
                        (self.best_score.get(flow, None) is None or
                            self.compare(ckpt["meta"][flow][self.save_best], self.best_score[flow])):
                    self.best_score[flow] = ckpt["meta"][flow][self.save_best]
                    self.best_ckpt_path[flow] = path

            if self.best_score.get(flow, None) is not None:
                runner.logger.info(
                    f"Found the best ckpt at \"{self.best_ckpt_path[flow]}\".")

            for path in paths:
                if path != self.best_ckpt_path.get(flow, None):
                    os.remove(path)

        if self.best_score.get(flow, None) is None or \
                self.compare(runner.log_buffer.output[self.save_best], self.best_score[flow]):
            self.best_score[flow] = runner.log_buffer.output[self.save_best]
            prev_best_ckpt_path = self.best_ckpt_path.get(flow, None)
            self.best_ckpt_path[flow] = osp.join(
                runner.work_dir,
                f"{flow}_best_{self.save_best}_epoch_{runner.epoch}.pth"
            )
            self._save_checkpoint(
                runner.work_dir,
                model.state_dict(),
                runner.optimizer,
                runner.lr_scheduler,
                runner.epoch,
                {flow: runner.log_buffer.output},
                f"{flow}_best_{self.save_best}"
                "_epoch_{}.pth"
            )
            if prev_best_ckpt_path is not None:
                os.remove(prev_best_ckpt_path)
