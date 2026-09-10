import torch as th
import math
from loguru import logger
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from dataset.utils import Normalizer


class VAE(nn.Module):
    def __init__(
        self,
        input_width=262,
        latent_dim=128,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        width=1024,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        clip_range=[-30, 20],
        normalizer=None,
        latent_normalizer=None,
    ):
        super().__init__()
        assert normalizer is not None, "normalizer is required"
        self.normalizer = normalizer
        self.latent_normalizer = latent_normalizer
        self.input_width: int = input_width
        self.decode_proj = nn.Linear(latent_dim, hidden_size)
        self.hidden_size = hidden_size
        self.encoder = CausalEncoder(
            input_width,
            hidden_size,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
            latent_dim=latent_dim,
            clip_range=clip_range,
        )
        self.decoder = CausalDecoder(
            input_width,
            hidden_size,
            down_t,
            stride_t,
            width,
            depth,
            dilation_growth_rate,
            activation=activation,
            norm=norm,
        )

    def preprocess(self, x, semantic_idx=0):
        x = self.normalizer.forward(x, semantic_idx)
        x = rearrange(x, "b l d -> b d l")
        return x

    def postprocess(self, x, semantic_idx=0):
        x = rearrange(x, "b d l -> b l d")
        x = self.normalizer.backward(x, semantic_idx)
        return x

    def encode(self, x, semantic_idx: int | th.Tensor = 0, return_mu_logvar=False) -> th.Tensor:
        if isinstance(semantic_idx, int):
            semantic_idx = th.ones(x.shape[0], dtype=th.long) * semantic_idx
        x_in = self.preprocess(x, semantic_idx)
        z, mu, logvar = self.encoder(x_in)
        if self.latent_normalizer is not None:
            z = self.latent_normalizer.forward(z)
        if return_mu_logvar:
            return z, mu, logvar
        return z

    def decode(self, z, semantic_idx=0) -> th.Tensor:
        if isinstance(semantic_idx, int):
            semantic_idx = th.ones(z.shape[0], dtype=th.long) * semantic_idx
        if self.latent_normalizer is not None:
            z = self.latent_normalizer.backward(z)
        z = self.decode_proj(z)
        x = self.decoder(z)  # (B, Cin, T)
        x = self.postprocess(x, semantic_idx)
        return x

    def forward(self, x, semantic_idx=0, loss_cfg=None):
        if isinstance(semantic_idx, int):
            semantic_idx = th.ones(x.shape[0], dtype=th.long) * semantic_idx
        x_in = self.preprocess(x, semantic_idx)
        z, mu, logvar = self.encoder(x_in)

        x_rec = self.decode_proj(z)
        x_rec = self.decoder(x_rec)  # (B, Cin, T)
        x_out = self.postprocess(x_rec, semantic_idx)

        if loss_cfg is not None:
            latent_gt = rearrange(x_in, "b d l -> b l d")
            latent_pred = rearrange(x_rec, "b d l -> b l d")
            loss = calculate_loss(
                latent_pred, latent_gt, x_out, x, mu, logvar, semantic_idx, loss_cfg
            )
            return x_out, loss

        return x_out


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,  # no padding here
            dilation=dilation,
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))  # only pad on the left
        return self.conv(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        input_emb_width=272,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        width=1024,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
        latent_dim=16,
        clip_range=[],
    ):
        super().__init__()
        self.clip_range: list[float] = clip_range
        self.proj = nn.Linear(width, latent_dim * 2)

        blocks = []
        filter_t, _ = stride_t * 2, stride_t // 2

        blocks.append(CausalConv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                CausalConv1d(input_dim, width, filter_t, stride_t, 1),
                CausalResnet1D(
                    width, depth, dilation_growth_rate, activation=activation, norm=norm
                ),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, hidden_size, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def forward(self, x) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        x = self.model(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        mu, logvar = x.chunk(2, dim=2)
        logvar = th.clamp(logvar, self.clip_range[0], self.clip_range[1])
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_emb_width=272,
        hidden_size=1024,
        down_t=2,
        stride_t=2,
        width=1024,
        depth=3,
        dilation_growth_rate=3,
        activation="relu",
        norm=None,
    ):
        super().__init__()
        blocks = []

        # filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(CausalConv1d(hidden_size, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                CausalResnet1D(
                    width,
                    depth,
                    dilation_growth_rate,
                    reverse_dilation=True,
                    activation=activation,
                    norm=norm,
                ),
                nn.Upsample(scale_factor=2, mode="nearest"),
                CausalConv1d(width, out_dim, 3, 1, 1),
            )
            blocks.append(block)
        blocks.append(CausalConv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(CausalConv1d(width, input_emb_width, 3, 1, 1))

        self.model = nn.Sequential(*blocks)

    def forward(self, z):
        z = z.transpose(1, 2)
        return self.model(z)


class nonlinearity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # swish
        return x * th.sigmoid(x)


class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=None):
        super().__init__()
        padding = dilation
        self.norm: str | None = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)

        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()

        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()

        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.conv1 = nn.Conv1d(n_in, n_state, 3, 1, padding, dilation)
        self.conv2 = nn.Conv1d(
            n_state,
            n_in,
            1,
            1,
            0,
        )

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1))
            x = self.activation1(x.transpose(-2, -1))
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1))
            x = self.activation2(x.transpose(-2, -1))
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class Resnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = [
            ResConv1DBlock(
                n_in, n_in, dilation=dilation_growth_rate**depth, activation=activation, norm=norm
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class CausalResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation="silu", norm=None, dropout=None):
        super().__init__()
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=n_in, eps=1e-6, affine=True)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
            self.norm2 = nn.BatchNorm1d(num_features=n_in, eps=1e-6, affine=True)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = nonlinearity()
            self.activation2 = nonlinearity()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()

        self.left_padding = (3 - 1) * dilation

        self.conv1 = nn.Conv1d(n_in, n_state, kernel_size=3, stride=1, padding=0, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_in, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x_orig = x
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
            x = self.activation1(x)
        else:
            x = self.norm1(x)
            x = self.activation1(x)

        x = nn.functional.pad(x, (self.left_padding, 0))

        x = self.conv1(x)

        if self.norm == "LN":
            x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
            x = self.activation2(x)
        else:
            x = self.norm2(x)
            x = self.activation2(x)

        x = self.conv2(x)
        x = x + x_orig
        return x


class CausalResnet1D(nn.Module):
    def __init__(
        self,
        n_in,
        n_depth,
        dilation_growth_rate=1,
        reverse_dilation=True,
        activation="relu",
        norm=None,
    ):
        super().__init__()

        blocks = [
            CausalResConv1DBlock(
                n_in, n_in, dilation=dilation_growth_rate**depth, activation=activation, norm=norm
            )
            for depth in range(n_depth)
        ]
        if reverse_dilation:
            blocks = blocks[::-1]

        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


LOG2PI = math.log(2 * math.pi)

loss_fn = {}


def register_loss_fn(key):
    def decorator(fn):
        loss_fn[key] = fn
        return fn

    return decorator


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * th.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * LOG2PI


def l1(x, y):
    return F.l1_loss(x, y, reduction="none").sum(dim=(1, 2))


def softclip(tensor, min):
    return min + F.softplus(tensor - min)


@register_loss_fn("kl")
def kl_loss(**kwargs):
    mu, logvar = kwargs["mu"], kwargs["logvar"]
    return -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=(1, 2)), None


def _recon_loss(gt, pred):
    log_sigma = ((gt - pred) ** 2).mean([0, 1, 2], keepdim=True).sqrt().log()
    log_sigma = softclip(log_sigma, -6)
    return gaussian_nll(pred, log_sigma, gt).sum(dim=(1, 2))


@register_loss_fn("latent")
def latent_loss(**kwargs):
    return _recon_loss(kwargs["latent_gt"], kwargs["latent_pred"]), None


@register_loss_fn("root")
def root_loss(**kwargs):
    return _recon_loss(
        kwargs["latent_gt"][..., : kwargs["root_dim"]],
        kwargs["latent_pred"][..., : kwargs["root_dim"]],
    ), None

@register_loss_fn("root_inverse")
def root_inverse_loss(**kwargs):
    return _recon_loss(
        kwargs["latent_gt"][..., - kwargs["root_dim"]:],
        kwargs["latent_pred"][..., - kwargs["root_dim"]:],
    ), None


def calculate_loss(
    latent_pred, latent_gt, recon_pred, recon_gt, mu, logvar, semantic_idx, loss_cfg
):
    kwargs = {
        "latent_pred": latent_pred,
        "latent_gt": latent_gt,
        "recon_pred": recon_pred,
        "recon_gt": recon_gt,
        "mu": mu,
        "logvar": logvar,
        "root_dim": loss_cfg.root_dim,
        "position_dim": loss_cfg.position_dim,
        "semantic_idx": semantic_idx,
    }
    loss = {"loss": 0.0}
    for key, weight in loss_cfg.kv.items():
        value, mask = loss_fn[key](**kwargs)
        if mask is None:
            mask = th.ones(value.shape[0], device=value.device, dtype=th.bool)
        mask = mask.to(value.device)
        if loss_cfg.get("mean_loss", False) or key == "kl":
            divisor = mask.sum() + 1.0e-8
        else:
            divisor = 1.0
        loss[key] = (value * mask).sum() / divisor
        loss["loss"] = loss["loss"] + weight * ((value * mask).sum() / divisor)
    return loss


def build_vae(
    vae_cfg: OmegaConf | dict,
    normalizer: Normalizer | None = None,
    latent_normalizer: Normalizer | None = None,
    checkpoint: str = "",
) -> VAE:
    vae = VAE(
        input_width=vae_cfg.input_width,
        latent_dim=vae_cfg.latent_dim,
        hidden_size=vae_cfg.hidden_size,
        down_t=vae_cfg.down_t,
        stride_t=vae_cfg.stride_t,
        width=vae_cfg.width,
        depth=vae_cfg.depth,
        dilation_growth_rate=vae_cfg.dilation_growth_rate,
        activation=vae_cfg.activation,
        norm=vae_cfg.norm,
        clip_range=vae_cfg.clip_range,
        normalizer=normalizer,
        latent_normalizer=latent_normalizer if vae_cfg.get("scale_latent", False) else None,
    )
    if checkpoint == "":
        return vae

    ckpt = th.load(checkpoint, map_location="cpu")
    if "epoch" in ckpt and "iter" in ckpt:
        logger.info(
            f"Loading checkpoint from {checkpoint}, epoch {ckpt['epoch']}, iter {ckpt['iter']}"
        )
    if "net" in ckpt:
        vae.load_state_dict(ckpt["net"], strict=True)
    elif "tae_model" in ckpt:
        vae.load_state_dict(ckpt["tae_model"], strict=True)
    else:
        raise ValueError(f"Checkpoint {checkpoint} does not contain 'net' or 'tae_model'")
    return vae
