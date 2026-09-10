from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, SequentialLR, LinearLR
from trainer.base import BaseTrainer
import warnings
from loguru import logger
from dataset import build_dataset
from dataset.utils import Normalizer
from model import build_vae
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")


class Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ema_decay = cfg.trainer.ema_decay
        self.lr = cfg.trainer.lr
        self.logger.info(f"Using {cfg.trainer.dataset_module} module")
        train_dataset = build_dataset[cfg.trainer.dataset_module](
            window_size=cfg.trainer.window_size,
            window_stride=cfg.trainer.window_stride,
            mode="train",
            feature_type=cfg.model.vae.feature_type,
            random_rotate=cfg.trainer.random_rotate,
            split_features=cfg.model.vae.split_features,
        )
        val_dataset = build_dataset[cfg.trainer.dataset_module](
            window_size=cfg.trainer.window_size,
            window_stride=cfg.trainer.window_stride,
            mode="val",
            feature_type=cfg.model.vae.feature_type,
            random_rotate=cfg.trainer.random_rotate,
            split_features=cfg.model.vae.split_features,
        )
        self.train_loader, self.val_loader = self.get_dataloader(train_dataset, val_dataset)
        self.set_key_iters(cfg.trainer.max_epoch, len(self.train_loader))
        milestones = [int(self.total_iters * 0.7), int(self.total_iters * 0.85)]
        self.update_cfg("trainer.milestones", milestones)

        self.vae = build_vae[cfg.model.vae.name](
            cfg.model.vae,
            Normalizer(cfg.model.vae.mean_fpath, cfg.model.vae.std_fpath),
        )
        self.forward = self.forward_unimodal
        self.vae.train()
        self.vae.to(self.device)
        self.logger.info(f"Total parameters: {self.total_parameters}")
        self.optimizer = AdamW(
            self.vae.parameters(),
            lr=cfg.trainer.lr,
            betas=(0.9, 0.99),
            weight_decay=cfg.trainer.weight_decay,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            [
                LinearLR(
                    self.optimizer,
                    start_factor=1 / cfg.trainer.warm_up_iter,
                    end_factor=1.0,
                    total_iters=cfg.trainer.warm_up_iter,
                ),
                MultiStepLR(
                    self.optimizer,
                    milestones=milestones,
                    gamma=cfg.trainer.gamma,
                ),
            ],
            milestones=[cfg.trainer.warm_up_iter],
        )

    def forward_unimodal(self, batch):
        _, semantic_idxs, batch = batch
        motions = batch.detach().float().cuda()
        _, loss = self.vae(motions, semantic_idxs, loss_cfg=self.cfg.trainer.loss)
        return loss


    @property
    def net(self):
        return self.vae


if __name__ == "__main__":
    cfg = OmegaConf.merge(
        OmegaConf.load("config/base.yaml"), OmegaConf.load("config/vae.yaml"), OmegaConf.from_cli()
    )
    OmegaConf.update(cfg, "project_name", "vae")

    if Path(cfg.save_model_dir).exists():
        if cfg.get("resume", False):
            logger.info(f"Resuming from {cfg.exp_root_dir}")
            resume_cfg = OmegaConf.load(Path(cfg.exp_root_dir) / "config.yaml")
            cfg = OmegaConf.merge(cfg, resume_cfg)
        else:
            raise ValueError(f"Save model directory {cfg.save_model_dir} already exists")
    else:
        Path(cfg.save_model_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.train()
