import numpy as np
import time
import torch as th
import torch.nn as nn
from .BERT.BERT_encoder import load_bert
import math
from einops import rearrange, repeat
from .transport import create_transport
from loguru import logger
from typing import Optional

class DiT(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_mode,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0,
        clip_dim=512,
        cond_mask_prob=0.1,
        num_frame_per_block=0,  # 0 means no causal mask
        max_length=75,
        clip_noise=20.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_frame_per_block = num_frame_per_block
        self.cond_mode = cond_mode
        self.cond_mask_prob = cond_mask_prob
        self.max_length = max_length
        self.clip_noise = clip_noise
        self.t_embedder = TimestepEmbedder(self.latent_dim)
        self.x_embedder = nn.Linear(self.input_dim, self.latent_dim)

        self.freqs = rope_params(1024, self.latent_dim // self.num_heads)
        self.blocks = nn.ModuleList(
            [
                AttentionBlockwithAdaLN(
                    self.latent_dim,
                    self.ff_size,
                    self.num_heads,
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.cond_mode == "text":
            self.y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.final_layer = nn.Linear(self.latent_dim, self.input_dim)
        self.initialize_weights()

        if self.cond_mode == "text":
            logger.info("Loading CLIP...")
            self.clip_model = self.load_and_freeze_clip()

        self.diffusion = create_transport(num_frame_per_block=self.num_frame_per_block)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                th.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith("clip_model.")]

    def load_and_freeze_clip(self):
        bert_model_path = "distilbert/distilbert-base-uncased"
        clip_model = load_bert(bert_model_path)
        return clip_model

    def mask_cond(self, cond, force_mask=False):
        t, bs, d = cond.shape
        if force_mask:
            return th.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = th.bernoulli(th.ones(bs, device=cond.device) * self.cond_mask_prob).view(
                1, bs, 1
            )
            return cond * (1.0 - mask)
        else:
            return cond

    @th.no_grad()
    def encode_text(self, raw_text):
        enc_text, mask = self.clip_model(raw_text)
        enc_text = enc_text.permute(1, 0, 2)
        return enc_text, mask

    def encode_cond(self, conds):
        if self.cond_mode == "text":
            return self.encode_text(conds)
        else:
            raise NotImplementedError("Unsupported condition mode.")

    def forward(self, x, t, conds, conds_mask, attn_mask, force_mask=False):
        conds = self.mask_cond(conds, force_mask=force_mask)
        t = self.t_embedder(t, dtype=x.dtype)
        x_embed = self.x_embedder(x)
        conds = self.y_embedder(conds)
        conds = rearrange(conds, "len batch dim -> batch len dim", batch=x.shape[0])
        conds_mask = conds_mask[:, None, None]
        self.freqs = self.freqs.to(x.device)
        for block in self.blocks:
            x_embed = block(
                x_embed,
                t,
                freqs=self.freqs,
                context=conds,
                attention_mask=attn_mask,
                context_mask=conds_mask,
            )
        return self.final_layer(x_embed)

    def forward_loss(self, latents, conds, m_lens, symmetric_t=False):
        non_pad_mask = generate_non_pad_mask(m_lens, self.max_length)
        is_two_persons = m_lens[:, 1] > 0
        target = th.where(rearrange(non_pad_mask, "b l -> b l 1"), latents, th.zeros_like(latents))
        cond_vector, conds_mask = self.encode_cond(conds)
        attn_mask = generate_attention_mask(
            non_pad_mask, self.num_frame_per_block, is_two_persons=is_two_persons
        )
        model_kwargs = dict(conds=cond_vector, attn_mask=attn_mask, conds_mask=conds_mask)
        loss_dict = self.diffusion.training_losses(
            self.forward, target, model_kwargs, dim=(2,), symmetric_t=symmetric_t
        )
        loss = loss_dict["loss"]
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum()
        return loss

    def denoise(
        self,
        x,
        cond_vector,
        conds_mask,
        attn_mask,
        from_noise_levels,
        to_noise_levels,
        cfg=3.0,
        context1=None,
        context2=None,
    ):
        x = x.clone()
        pred_v = self.forward(x, from_noise_levels, cond_vector, conds_mask, attn_mask)
        if not cfg == 1.0:
            cond_eps, uncond_eps = th.chunk(pred_v, 2, dim=0)
            half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
            pred_v = th.cat([half_eps, half_eps], dim=0)
        if context1 is None:
            delta = to_noise_levels - from_noise_levels
            pred_x = x + delta.unsqueeze(-1) * pred_v
        else:
        # if True:
            pred_x0 = x + (1 - from_noise_levels).unsqueeze(-1) * pred_v

            if context1 is not None:
                if not cfg == 1.0:
                    context1 = repeat(context1, "b l d -> (2 b) l d")
                pred_x0[:, : context1.shape[1]] = context1
            if context2 is not None:
                if not cfg == 1.0:
                    context2 = repeat(context2, "b l d -> (2 b) l d")
                pred_x0[:, self.max_length : self.max_length + context2.shape[1]] = context2

            new_noise = th.randn_like(pred_x0)
            pred_x = pred_x0 * to_noise_levels.unsqueeze(-1) + new_noise * (
                1 - to_noise_levels
            ).unsqueeze(-1)
        if not cfg == 1.0:
            pred_x = th.chunk(pred_x, 2, dim=0)[0]
        return pred_x

    def generate(
        self,
        conds,
        m_lens,
        cond_scale=2.5,
        scheduling_matrix_type="full_sequence",
        sampling_timesteps=50,
        uncertainty_scale=1.0,
        history_length=10,
        context1=None,
        context2=None,
        n_persons=2,
        return_time=False,
    ):
        device = next(self.parameters()).device
        m_lens_original = m_lens.clone()
        curr_L = 0 if context1 is None else context1.shape[1]
        m_lens = m_lens + curr_L
        B, L = len(m_lens), max(m_lens.flatten())
        x1 = th.zeros((B, L, self.input_dim), device=device).float()
        x2 = th.zeros((B, L, self.input_dim), device=device).float()
        h1, h2 = None, None

        time_all = 0.0

        return_person2 = False
        if context1 is None and context2 is not None:
            context1 = context2
            return_person2 = True
        if context1 is not None:
            x1[:, :curr_L] = context1
            with th.no_grad():
                h1 = context1[:, -history_length:].clone()
        if context2 is not None:
            x2[:, :curr_L] = context2
            with th.no_grad():
                h2 = context2[:, -history_length:].clone()
        non_pad_mask = generate_non_pad_mask(m_lens, max(self.max_length, L))
        non_pad_mask = rearrange(non_pad_mask, "batch (n len) -> batch n len", n=2)
        logger.debug(f"History length: {history_length}")
        start_time = time.time()
        cond_vector, cond_mask = self.encode_cond(conds)
        time_each_block = 0.0
        if not cond_scale == 1.0:
            cond_vector = th.cat([cond_vector, th.zeros_like(cond_vector)], dim=1)
            cond_mask = th.cat([cond_mask, cond_mask], dim=0)
        time_all += time.time() - start_time

        while curr_L < L:
            start_frame = max(0, curr_L - history_length)
            end_frame = min(L, start_frame + self.max_length)

            logger.debug(f"start_frame: {start_frame} | end_frame: {end_frame}")

            curr_non_pad_mask = non_pad_mask[:, :, start_frame:end_frame]
            if curr_non_pad_mask.shape[2] < self.max_length:
                padding = th.zeros(
                    (B, 2, self.max_length - curr_non_pad_mask.shape[2]), dtype=th.bool
                )
                curr_non_pad_mask = th.cat([curr_non_pad_mask, padding], dim=2)
            curr_non_pad_mask = rearrange(curr_non_pad_mask, "batch n len -> batch (n len)")
            is_two_persons = m_lens_original[:, 1] > 0
            start_time = time.time()
            attn_mask = generate_attention_mask(
                curr_non_pad_mask, self.num_frame_per_block, is_two_persons=is_two_persons
            )
            if not cond_scale == 1.0:
                attn_mask = th.cat([attn_mask, attn_mask], dim=0)
            time_all += time.time() - start_time

            tgt_length = min(L, self.max_length, end_frame - start_frame)
            logger.debug(f"tgt_length: {tgt_length}")
            scheduling_matrix = (
                generate_scheduling_matrix(
                    tgt_length,
                    self.max_length,
                    self.num_frame_per_block,
                    sampling_timesteps,
                    uncertainty_scale,
                    scheduling_matrix_type,
                )
                .to(device)
                .float()
            )
            start_time = time.time()
            xs_pred = th.zeros((B, 2, self.max_length, self.input_dim), device=device).float()
            xs_pred[:, :, :tgt_length] = th.randn((B, 2, tgt_length, self.input_dim), device=device)
            xs_pred = th.clamp(xs_pred, -self.clip_noise, self.clip_noise)
            xs_pred = rearrange(xs_pred, "b n l d -> b (n l) d")
            time_all += time.time() - start_time
            logger.debug(f"Time taken before denoise: {time_all}")
            time_denoise = 0.0
            for m in range(scheduling_matrix.shape[0] - 1):
                xs_pred = rearrange(xs_pred, "b (n l) d -> b n l d", n=2)
                from_noise_levels = repeat(scheduling_matrix[m], "l -> b (n l)", b=B, n=2)
                to_noise_levels = repeat(scheduling_matrix[m + 1], "l -> b (n l)", b=B, n=2)
                xs_pred = rearrange(xs_pred, "b n l d -> b (n l) d")

                start_time = time.time()
                if not cond_scale == 1.0:
                    from_noise_levels = repeat(from_noise_levels, "b l -> (2 b) l")
                    to_noise_levels = repeat(to_noise_levels, "b l -> (2 b) l")
                    xs_pred = repeat(xs_pred, "b l d -> (2 b) l d")
                time_all += time.time() - start_time
                time_denoise += time.time() - start_time

                current_xs_pred = xs_pred.clone()
                start_time = time.time()
                next_xs_pred = self.denoise(
                    current_xs_pred,
                    cond_vector=cond_vector,
                    conds_mask=cond_mask,
                    attn_mask=attn_mask,
                    from_noise_levels=from_noise_levels,
                    to_noise_levels=to_noise_levels,
                    cfg=cond_scale,
                    context1=h1,
                    context2=h2,
                )
                time_all += time.time() - start_time
                time_denoise += time.time() - start_time
                xs_pred = next_xs_pred.clone()
                if m == sampling_timesteps - 1:
                    time_first_frame = time_all
                if m == sampling_timesteps + uncertainty_scale - 1:
                    time_each_block = time_all - time_first_frame
            xs_pred = rearrange(xs_pred, "b (n l) d -> b n l d", n=2)
            logger.debug(f"Time taken for denoise: {time_denoise}")

            if n_persons == 2:
                x1[:, start_frame:end_frame] = xs_pred[:, 0, :tgt_length].clone()
                x2[:, start_frame:end_frame] = xs_pred[:, 1, :tgt_length].clone()
                h1 = x1[:, end_frame - history_length : end_frame].clone()
                h2 = x2[:, end_frame - history_length : end_frame].clone()
            if n_persons == 1:
                x1[:, start_frame:end_frame] = xs_pred[:, 0, :tgt_length].clone()
                h1 = x1[:, end_frame - history_length : end_frame].clone()
            curr_L = end_frame
        if return_time:
            if n_persons == 2:
                return x1, x2, time_first_frame, time_each_block, time_all
            else:
                return x1, None, time_first_frame, time_each_block, time_all
        if n_persons == 2:
            return x1, x2
        if n_persons == 1:
            return x1, None

def generate_non_pad_mask(lengths: th.Tensor, max_length: int) -> th.Tensor:
    """Generate non-padding mask for the input sequence.

    Args:
        lengths (B, N) or (B,): lengths of the input sequences
        max_length (int): maximum length of the input sequence
    Returns:
        non_pad_mask (B, N * max_length): non-padding mask
    """
    if lengths.ndim == 1:
        non_pad_mask = lengths_to_mask(lengths, max_length)
        pad_mask = th.zeros_like(non_pad_mask)
        return th.cat([non_pad_mask, pad_mask], dim=1)
    else:
        non_pad_mask = [lengths_to_mask(lengths[:, i], max_length) for i in range(lengths.shape[1])]
        non_pad_mask = th.cat(non_pad_mask, dim=1)
        return non_pad_mask


def build_causal_mask(lq: int, num_frame_per_block: int) -> th.Tensor:
    """Build causal mask for the input sequence.

    Args:
        lq (int): length of the input sequence
        num_frame_per_block (int): number of frames per block

    Returns:
        causal_mask (L, L): causal mask
    """
    total_length = lq
    ends = th.zeros(total_length, dtype=th.long)
    frame_indices = th.arange(start=0, end=total_length, step=num_frame_per_block)
    for tmp in frame_indices:
        ends[tmp : tmp + num_frame_per_block] = tmp + num_frame_per_block

    def attention_mask(q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    q_idx = th.arange(lq).unsqueeze(1)
    k_idx = th.arange(lq).unsqueeze(0)

    causal_mask = attention_mask(q_idx, k_idx)
    return causal_mask


def generate_attention_mask(
    non_pad_mask: th.Tensor, num_frame_per_block: int = 0, is_two_persons: Optional[th.Tensor] = None
) -> th.Tensor:
    """Generate attention mask for the input sequence.

    Args:
        non_pad_mask (B, N * max_length): non-padding mask
        num_frame_per_block (int): number of frames per block

    Returns:
        attn_mask (B, 1, N * max_length, N * max_length): attention mask
    """
    device = non_pad_mask.device
    L = non_pad_mask.shape[1]
    if num_frame_per_block > 0:
        L = L // 2
        attn_mask_two_persons = build_causal_mask(L, num_frame_per_block).to(device)
        attn_mask_two_persons = th.cat(
            [
                th.cat([attn_mask_two_persons, attn_mask_two_persons], dim=1),
                th.cat([attn_mask_two_persons, attn_mask_two_persons], dim=1),
            ],
            dim=0,
        )
        attn_mask_one_person = build_causal_mask(L, num_frame_per_block).to(device)
        attn_mask_one_person = th.cat(
            [
                th.cat([attn_mask_one_person, th.zeros_like(attn_mask_one_person)], dim=1),
                th.cat(
                    [th.zeros_like(attn_mask_one_person), th.zeros_like(attn_mask_one_person)],
                    dim=1,
                ),
            ]
        )
        if is_two_persons is None:
            is_two_persons = th.ones(non_pad_mask.shape[0], dtype=th.bool, device=device)
        assert is_two_persons.shape[0] == non_pad_mask.shape[0]
        attn_mask = th.where(
            is_two_persons[:, None, None], attn_mask_two_persons, attn_mask_one_person
        )[:, None]
        # attn_mask = attn_mask_two_persons[:, None]
    else:
        attn_mask = th.ones(L, L, dtype=th.bool, device=device)[None, None]
    # return attn_mask[None, None] & non_pad_mask[:, None, None]
    return attn_mask & (non_pad_mask.unsqueeze(2) * non_pad_mask.unsqueeze(1))[:, None]
    # return attn_mask & non_pad_mask[:, None, None]


def generate_pyramid_scheduling_matrix(
    sampling_timesteps, horizon: int, uncertainty_scale: float = 1.0
):
    height = sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
    scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
    for m in range(height):
        for t in range(horizon):
            scheduling_matrix[m, t] = sampling_timesteps + int(t * uncertainty_scale) - m
    return np.clip(scheduling_matrix, 0, sampling_timesteps)


def generate_full_sequence_scheduling_matrix(
    sampling_timesteps, horizon: int, uncertainty_scale: float = 1.0
):
    return np.arange(sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)


def generate_autoregressive_scheduling_matrix(
    sampling_timesteps, horizon: int, uncertainty_scale: float = 1.0
):
    return generate_pyramid_scheduling_matrix(sampling_timesteps, horizon, sampling_timesteps)


def generate_scheduling_matrix(
    tgt_length,
    max_length,
    num_frame_per_block,
    sampling_timesteps,
    uncertainty_scale,
    scheduling_matrix_type,
):
    scheduling_matrix_func = {
        "pyramid": generate_pyramid_scheduling_matrix,
        "full_sequence": generate_full_sequence_scheduling_matrix,
        "autoregressive": generate_autoregressive_scheduling_matrix,
    }
    if num_frame_per_block == 0:
        num_frame_per_block = 1
        scheduling_matrix_type = "full_sequence"
    n_blocks = max(1, tgt_length // num_frame_per_block)
    if tgt_length % num_frame_per_block != 0:
        n_blocks += 1
    scheduling_matrix = scheduling_matrix_func[scheduling_matrix_type](
        sampling_timesteps, n_blocks, uncertainty_scale
    )
    scheduling_matrix = scheduling_matrix.repeat(num_frame_per_block, axis=1)
    scheduling_matrix = scheduling_matrix[:, :tgt_length]
    scheduling_matrix = 1 - scheduling_matrix / sampling_timesteps
    padding_length = max_length - tgt_length
    padding_scheduling_matrix = np.zeros((scheduling_matrix.shape[0], padding_length))
    scheduling_matrix = np.concatenate([scheduling_matrix, padding_scheduling_matrix], axis=1)
    return th.from_numpy(scheduling_matrix).float()


def build_dit(dit_cfg, checkpoint="") -> DiT:
    layer = dit_cfg.layer
    dit = DiT(
        latent_dim=max(layer * 64, 512),
        ff_size=max(layer * 64 * 2, 1024),
        num_layers=layer,
        num_heads=layer // 2,
        dropout=0,
        clip_dim=768,
        cond_mask_prob=0.1,
        num_frame_per_block=dit_cfg.num_frame_per_block,
        max_length=dit_cfg.max_length,
        clip_noise=dit_cfg.clip_noise,
        cond_mode=dit_cfg.cond_mode,
        input_dim=dit_cfg.input_dim,
    )
    if checkpoint == "":
        return dit
    ckpt = th.load(checkpoint, map_location="cpu")
    logger.info(f"Loading checkpoint from {checkpoint}, epoch {ckpt['epoch']}, iter {ckpt['iter']}")
    dit.load_state_dict(ckpt["net_ema"] if "net_ema" in ckpt else ckpt["net"], strict=True)
    return dit


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=th.float32):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        if t.dim() == 1:
            t = t.unsqueeze(1)

        B, L = t.shape
        t = t.reshape(B * L)
        half = dim // 2
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=dtype) / half).to(
            device=t.device, dtype=dtype
        )
        args = t[:, None] * freqs[None]
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        embedding = embedding.reshape(B, L, -1)
        return embedding

    def forward(self, t, dtype=th.bfloat16):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


def lengths_to_mask(lengths, max_len):
    mask = th.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask  # (b, len)


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    # compute in float32 for speed; returns complex64
    inv = th.arange(0, dim, 2, dtype=th.float32) / float(dim)
    base = th.tensor(float(theta), dtype=th.float32)
    freqs = th.outer(th.arange(max_seq_len, dtype=th.float32), 1.0 / th.pow(base, inv))
    freqs = th.polar(th.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, freqs):
    n, c = x.size(2), x.size(3) // 2

    # fast path: vectorized apply up to max length, then restore padding per sample
    # cast once to complex for whole batch (use float32/complex64 for speed)
    dtype_orig = x.dtype
    max_len = x.size(1)
    xb = x
    xc = th.view_as_complex(
        xb.to(th.float32).reshape(xb.size(0), max_len, n, -1, 2)
    )  # (B, Fmax, N, C/2)

    # slice freqs to max_len and broadcast over batch/heads
    freqs_i = freqs[:max_len].to(xc.dtype).view(max_len, 1, -1)  # (Fmax, 1, C)

    # apply rotary embedding (time-only) in one go
    yc = xc * freqs_i  # (B, Fmax, N, C/2)
    yr = th.view_as_real(yc)
    yr = yr.flatten(3)  # (B, Fmax, N, C)
    return yr.to(dtype_orig)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(th.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * th.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


def attention(q, k, v, attention_mask, dtype=th.float):
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    attention_mask = attention_mask.to(q.device)
    out = th.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    out = out.transpose(1, 2).contiguous()
    return out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, attention_mask, freqs=None):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        x = attention(
            q=rope_apply(q, freqs) if freqs is not None else q,
            k=rope_apply(k, freqs) if freqs is not None else k,
            v=v,
            attention_mask=attention_mask,
        )

        x = x.flatten(2)
        x = self.o(x)
        return x


class T2VCrossAttention(SelfAttention):
    def forward(self, x, context, context_mask, freqs=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        q = rope_apply(q, freqs) if freqs is not None else q
        x = attention(q, k, v, attention_mask=context_mask)
        x = x.flatten(2)
        x = self.o(x)
        return x


class AttentionBlockwithAdaLN(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=True, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = LayerNorm(dim, eps)
        self.self_attn = SelfAttention(dim, num_heads, qk_norm, eps)
        self.cross_attn = T2VCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = (
            LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.norm2 = LayerNorm(dim, eps)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, 6 * self.dim, bias=True)
        )

    def forward(self, x, e, attention_mask, freqs, context, context_mask):
        e = (self.adaLN_modulation(e)).chunk(6, dim=-1)
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], attention_mask, freqs)
        x = x + y * e[2]
        x = x + self.cross_attn(self.norm3(x), context, context_mask)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x
