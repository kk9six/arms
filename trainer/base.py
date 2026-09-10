from pathlib import Path
import sys
from typing import Any
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from utils import seed_everything
from utils import get_logger
import json
import wandb
from collections import defaultdict, OrderedDict


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        seed_everything(cfg.seed)
        OmegaConf.save(cfg, Path(cfg.exp_root_dir) / "config.yaml")
        self.logger = get_logger(
            Path(cfg.exp_root_dir),
            cfg.log_file,
            cfg.log_level,
            "a" if cfg.get("resume", False) else "w",
        )
        self.wandb_run = False
        if self.cfg.log_level == "INFO":
            run = wandb.init(
                name=cfg.exp_name,
                project=cfg.project_name,
                config=OmegaConf.to_container(cfg, resolve=True),
                dir=cfg.exp_root_dir,
                id=cfg.get("wandb_id", None),
                resume="must" if cfg.get("resume", False) else None,
            )
            if not cfg.get("resume", False):
                self.update_cfg("wandb_id", run.id)
            self.wandb_run = True
        self.logger.info(" ".join([sys.executable] + sys.argv))
        self.logger.info(
            f"Config\n{json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=4)}"
        )
        self.resume_time = 0


    def set_key_iters(self, max_epoch: int, iter_per_epoch: int):
        self.iter_per_epoch = iter_per_epoch
        self.max_epoch = max_epoch
        self.total_iters = max_epoch * iter_per_epoch
        self.warm_up_iter = iter_per_epoch // 4
        self.print_iter = iter_per_epoch // 10
        self.save_iter = iter_per_epoch // 2
        self.backup_epoch = []
        if "backup_epoch" in self.cfg.trainer:
            self.backup_epoch = OmegaConf.to_object(self.cfg.trainer.backup_epoch)
            assert isinstance(self.backup_epoch, list), "backup_epoch must be a list"
            self.update_cfg("trainer.backup_epoch", self.backup_epoch)
        else:
            self.backup_epoch = [250]
            self.update_cfg("trainer.backup_epoch", self.backup_epoch)
        self.update_cfg("trainer.total_iters", self.total_iters)
        self.update_cfg("trainer.warm_up_iter", self.warm_up_iter)
        self.update_cfg("trainer.print_iter", self.print_iter)
        self.update_cfg("trainer.save_iter", self.save_iter)
        self.logger.info(
            f"Total epochs: {self.max_epoch}, total iterations: {self.total_iters}, {iter_per_epoch} iterations per epoch"
        )
        self.logger.info(f"Warm-up iterations: {self.warm_up_iter}")
        self.logger.info(f"Print every {self.print_iter} iterations")
        self.logger.info(f"Save every {self.save_iter} iterations")

    def update_cfg(self, key: str, value: Any) -> None:
        OmegaConf.update(self.cfg, key, value)
        OmegaConf.save(self.cfg, Path(self.cfg.exp_root_dir) / "config.yaml")
        self.logger.info(f"Updated config: {key} = {value}")

    def get_dataloader(
        self, train_dataset: Dataset, val_dataset: Dataset
    ) -> tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=True,
            num_workers=self.cfg.trainer.num_workers,
            drop_last=True,
            prefetch_factor=self.cfg.trainer.prefetch_factor,
            pin_memory=self.cfg.trainer.pin_memory,
            persistent_workers=self.cfg.trainer.persistent_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.trainer.num_workers,
            drop_last=True,
            prefetch_factor=self.cfg.trainer.prefetch_factor,
            pin_memory=self.cfg.trainer.pin_memory,
            persistent_workers=self.cfg.trainer.persistent_workers,
        )
        return train_loader, val_loader

    def update_ema(self):
        if self.ema_decay > 0:
            with torch.no_grad():
                for p_src, p_tgt in zip(self.net.parameters(), self.net_ema.parameters()):
                    p_tgt.data.mul_(self.ema_decay).add_(p_src.data, alpha=1 - self.ema_decay)

    def start_timer(self):
        self.start_time = time.time()

    def update_lr_warm_up(self, iter, warm_up_iter, lr):
        current_lr = lr * (iter + 1) / (warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    @property
    def time_since_start(self):
        return time.time() - self.start_time + self.resume_time

    def format_time(self, seconds: float) -> str:
        seconds = int(seconds)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    def log_wandb(self, log_data: dict, step: int):
        if self.wandb_run:
            wandb.log(log_data, step=step)

    def log_console(self, log_data: dict, inner_step: int, outer_step: int, epoch: int, task: str):
        completed_percent = outer_step / self.total_iters
        time_since_start = self.time_since_start
        time_remaining = (time_since_start / completed_percent) - time_since_start

        epoch_info = f"Epoch/Iter {epoch:03d}/{inner_step:05d} ({outer_step:07d})"
        time_info = f"{self.format_time(time_since_start)} (- {self.format_time(time_remaining)}), completed {completed_percent * 100:.2f}%"
        training_info = "\t".join([f"{k.capitalize()}: {v:.5f}" for k, v in log_data.items()])

        self.logger.info(f"{epoch_info} | {time_info} | {task} | {training_info}")

    @property
    def total_parameters(self) -> int:
        return f"{sum(param.numel() for param in self.net.parameters()) / 1_000_000:.3f}M"

    def save(self, n_iter, epoch, lr, name="latest"):
        torch.save(
            {
                "net": self.net.state_dict(),
                "net_ema": self.net_ema.state_dict() if self.ema_decay > 0 else None,
                "lr": lr,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "iter": n_iter,
                "time_since_start": self.time_since_start,
            },
            Path(self.cfg.save_model_dir) / f"{name}.pth",
        )

    def post_train(self):
        ...

    def train(self):
        iter = 0
        start_epoch = 1
        min_val_loss = np.inf

        self.logger.info(f"Resume: {self.cfg.get('resume', False)}")
        if self.cfg.get("resume", False):
            ckpt = torch.load(Path(self.cfg.save_model_dir) / "latest.pth")
            self.net.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.resume_time = ckpt["time_since_start"]
            start_epoch = ckpt["epoch"]
            iter = ckpt["iter"]
            self.logger.info(f"Resuming from {self.cfg.save_model_dir}/latest.pth")
            self.logger.info(f"Start epoch: {start_epoch}, start iter: {iter}")

        log_data = defaultdict(lambda: 0.0, OrderedDict())
        self.start_timer()
        for epoch in range(start_epoch, self.max_epoch + 1):
            self.net.train()
            for i, batch in enumerate(self.train_loader):
                iter += 1
                if iter <= self.warm_up_iter:
                    self.scheduler.step()
                loss_dict = self.forward(batch)
                self.optimizer.zero_grad()
                loss_dict["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                self.optimizer.step()
                if iter > self.warm_up_iter:
                    self.scheduler.step()
                self.update_ema()

                log_data["train/lr"] += self.optimizer.param_groups[0]["lr"]
                for k, v in loss_dict.items():
                    log_data[f"train/{k}"] += v.item()

                if iter % self.print_iter == 0:
                    for k, v in log_data.items():
                        log_data[k] /= self.print_iter
                    self.log_wandb(log_data, iter)
                    self.log_console(log_data, i + 1, iter, epoch, "Training")
                    log_data = defaultdict(lambda: 0.0, OrderedDict())

                if iter % self.save_iter == 0:
                    self.save(iter, epoch, self.optimizer.param_groups[0]["lr"], name="latest")

            if self.backup_epoch and epoch in self.backup_epoch:
                self.save(iter, epoch, self.optimizer.param_groups[0]["lr"], f"backup_{epoch}")

            self.net.eval()
            with torch.no_grad():
                val_loss_data = defaultdict(lambda: 0.0, OrderedDict())
                for batch in self.val_loader:
                    loss_dict = self.forward(batch)
                    for k, v in loss_dict.items():
                        val_loss_data[f"val/{k}"] += v.item()
                for k, v in val_loss_data.items():
                    val_loss_data[k] /= len(self.val_loader)
                self.log_wandb(val_loss_data, iter)
                self.log_console(val_loss_data, 0, iter, epoch, "Validation")
                if val_loss_data["val/loss"] < min_val_loss:
                    min_val_loss = val_loss_data["val/loss"]
                    self.save(iter, epoch, self.optimizer.param_groups[0]["lr"], name="best")
                    self.logger.info(f"New best validation loss: {val_loss_data['val/loss']:.5f}")

        if self.wandb_run:
            wandb.finish()

        self.post_train()
