from omegaconf import OmegaConf
from einops import rearrange
import copy
from pathlib import Path
from trainer.base import BaseTrainer
import torch as th
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, SequentialLR, LinearLR
from dataset import build_dataset
from dataset.utils import Normalizer
from model import build_vae
from model.dit import build_dit  # , generate_non_pad_mask


class Trainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ema_decay = cfg.trainer.ema_decay
        self.lr = cfg.trainer.lr
        feature_type = cfg.model.vae.feature_type if "vae" in cfg.model else cfg.model.feature_type
        train_dataset = build_dataset[cfg.trainer.dataset_module](
            mode="train",
            feature_type=feature_type,
            random_rotate=cfg.trainer.random_rotate,
            max_length=cfg.trainer.max_length,
        )
        val_dataset = build_dataset[cfg.trainer.dataset_module](
            mode="val",
            feature_type=feature_type,
            random_rotate=cfg.trainer.random_rotate,
            max_length=cfg.trainer.max_length,
        )
        self.train_loader, self.val_loader = self.get_dataloader(train_dataset, val_dataset)
        self.set_key_iters(cfg.trainer.max_epoch, len(self.train_loader))
        milestones = [
            int(self.total_iters * 0.5),
            int(self.total_iters * 0.7),
            int(self.total_iters * 0.85),
        ]
        self.update_cfg("trainer.dit.milestones", milestones)

        if "vae" in cfg.model:
            self.vae = build_vae[cfg.model.vae.name](
                cfg.model.vae,
                Normalizer(cfg.model.vae.mean_fpath, cfg.model.vae.std_fpath),
                None,
                checkpoint=Path(cfg.model.vae.path) / "model" / "latest.pth",
            )
            self.vae.eval()
            self.vae.to(self.device)
            self.logger.info(f"VAE loaded from {Path(cfg.model.vae.path) / 'latest.pth'}")
        else:
            self.logger.info("No VAE used")

        self.dit = build_dit(cfg.model.dit)
        self.dit.train()
        self.dit.to(self.device)
        self.logger.info(f"Total parameters: {self.total_parameters}")
        self.optimizer = optim.AdamW(
            self.dit.parameters(),
            lr=self.lr,
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

        if self.ema_decay > 0:
            self.net_ema = copy.deepcopy(self.dit)
            self.net_ema.eval()

        forward_func = {
            "unified_dataset": self.forward_unified_dataset,
            "symmetric_input": self.forward_symmetric_input,
            "interx": self.forward_interx,
        }
        self.forward = forward_func[cfg.trainer.forward_function]
        self.symmetric_t = cfg.trainer.symmetric_t

    def forward_unified_dataset(self, batch):
        conds, motion1, motion2, m_lens1, m_lens2 = batch
        motion1 = motion1.detach().float().cuda()  # [B, L, D]
        motion2 = motion2.detach().float().cuda()  # [B, L, D]
        m_lens1 = m_lens1.detach().long().cuda()
        m_lens2 = m_lens2.detach().long().cuda()
        with th.no_grad():
            latent1 = self.vae.encode(motion1, semantic_idx=0)
            latent2 = self.vae.encode(motion2, semantic_idx=1)
        latent = th.cat([latent1, latent2], dim=1)
        m_lens1 = m_lens1 // 4
        m_lens2 = m_lens2 // 4
        m_lens = th.stack([m_lens1, m_lens2], dim=-1)
        loss = self.dit.forward_loss(latent, conds, m_lens, symmetric_t=self.symmetric_t)
        return {"loss": loss}

    def forward_interx(self, batch):
        _, _, conds, _, motions, m_lens1, _ = batch
        motions = motions.detach().float().cuda()  # [B, L, D]
        motion1 = motions[:, :, 0]
        motion2 = motions[:, :, 1]
        with th.no_grad():
            latent1 = self.vae.encode(motion1, semantic_idx=0)
            latent2 = self.vae.encode(motion2, semantic_idx=1)
        latent = th.cat([latent1, latent2], dim=1)
        m_lens1 = m_lens1.detach().long().cuda() // 4
        m_lens = th.stack([m_lens1, m_lens1], dim=-1)
        loss = self.dit.forward_loss(latent, conds, m_lens, symmetric_t=self.symmetric_t)
        return {"loss": loss}

    def forward_symmetric_input(self, batch):
        conds, motion1, motion2, m_lens1, m_lens2 = batch
        motion1 = motion1.detach().float().cuda()  # [B, L, D]
        motion2 = motion2.detach().float().cuda()  # [B, L, D]
        m_lens1 = m_lens1.detach().long().cuda()
        m_lens2 = m_lens2.detach().long().cuda()
        with th.no_grad():
            latent1 = self.vae.encode(motion1, semantic_idx=0)
            latent2 = self.vae.encode(motion2, semantic_idx=1)
            latent2 = th.where(m_lens2[:, None, None] > 0, latent2, latent1)
        latent = th.cat([latent1, latent2], dim=1)
        m_lens1 = m_lens1 // 4
        m_lens = th.stack([m_lens1, m_lens1], dim=-1)
        loss = self.dit.forward_loss(latent, conds, m_lens, symmetric_t=self.symmetric_t)
        return {"loss": loss}

    @property
    def net(self):
        return self.dit


if __name__ == "__main__":
    cfg = OmegaConf.merge(
        OmegaConf.load("config/base.yaml"), OmegaConf.load("config/dit.yaml"), OmegaConf.from_cli()
    )
    OmegaConf.update(cfg, "project_name", "dit")
    if "vae" in cfg.model:
        vae_cfg = OmegaConf.load(Path(cfg.model.vae.path) / "config.yaml")
        OmegaConf.update(cfg, "model.vae", vae_cfg.model.vae)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    if Path(cfg.save_model_dir).exists():
        if cfg.get("resume", False):
            cfg = OmegaConf.load(Path(cfg.exp_root_dir) / "config.yaml")
        else:
            raise ValueError(f"Save model directory {cfg.save_model_dir} already exists")
    else:
        Path(cfg.save_model_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.train()
