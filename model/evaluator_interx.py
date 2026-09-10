from dataset.interx import interhuman_to_interx
from tqdm import tqdm
from loguru import logger
from model.vae import VAE
from model.dit import DiT
import time
from einops import repeat
import copy
import numpy as np
import os
from os.path import join as pjoin

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from utils.word_vectorizer import WordVectorizer, POS_enumerator
from dataset.utils import features_to_interhuman
from dataset.interx import interhuman_to_interx, collate_fn


class EvaluationDataset(Dataset):
    def __init__(
        self,
        vae: VAE,
        dit: DiT,
        dataset: Dataset,
        batch_size: int,
        mm_num_samples: int,
        mm_num_repeats: int,
        cond_scale: float,
        feature_type: str,
        history_length: int,
        sampling_timesteps: int,
        uncertainty_scale: float,
        scheduling_matrix_type: str,
        infer_from_velocity: bool,
    ):
        assert mm_num_samples < len(dataset)
        dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        self.w_vectorizer = dataset.w_vectorizer
        self.max_motion_length = dataset.max_motion_length

        generated_motion = []
        mm_generated_motions = []

        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        device = "cuda"

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating motions"):
                # name, text, motion1, motion2, motion_lens = data
                word_emb, pos_ohot, caption, cap_lens, motions, motion_lens, tokens = data

                motion1, motion2 = motions.split(motions.shape[-1] // 2, dim=-1)

                tokens = tokens[0].split("_")
                word_emb = word_emb.detach().to(device).float()
                pos_ohot = pos_ohot.detach().to(device).float()

                mm_num_now = len(mm_generated_motions)
                is_mm = (
                    True
                    if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now]))
                    else False
                )
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []

                for t in range(repeat_times):
                    ids_length = motion_lens.detach().long() // 4
                    ids_length = repeat(ids_length, "b -> b n", n=2)
                    latent1, latent2 = dit.generate(
                        caption,
                        ids_length,
                        cond_scale,
                        history_length=history_length,
                        sampling_timesteps=sampling_timesteps,
                        uncertainty_scale=uncertainty_scale,
                        scheduling_matrix_type=scheduling_matrix_type,
                        n_persons=2,
                    )

                    with torch.no_grad():
                        output1 = vae.decode(latent1, semantic_idx=0).detach().cpu()
                        output2 = vae.decode(latent2, semantic_idx=1).detach().cpu()

                    output_interx_list = []
                    for o1, o2 in zip(output1, output2):
                        o1_interhuman, o2_interhuman = features_to_interhuman(
                            o1,
                            o2,
                            feature_type=feature_type,
                            inference_from_velocity=infer_from_velocity,
                            local_dim=466
                        )
                        j1 = o1_interhuman[:, :22 * 3].reshape(-1, 22, 3).clone()
                        j2 = o2_interhuman[:, :22 * 3].reshape(-1, 22, 3).clone()
                        offset = j1[0, 0, 1].item()
                        j1[:, :, 1] = j1[:, :, 1] - offset
                        j2[:, :, 1] = j2[:, :, 1] - offset
                        o1_interhuman = torch.cat([j1.reshape(-1, 22 * 3), o1_interhuman[:, 22 * 3:]], dim=-1)
                        o2_interhuman = torch.cat([j2.reshape(-1, 22 * 3), o2_interhuman[:, 22 * 3:]], dim=-1)
                        o_interx = interhuman_to_interx(o1_interhuman, o2_interhuman)
                        # output_interx_list.append(torch.cat([o1.reshape(-1, 56, 6), o2.reshape(-1, 56, 6)], dim=-1))
                        output_interx_list.append(o_interx)
                    motion_output = torch.stack(output_interx_list)
                    gen_motion_len = motion_output.shape[1]

                    if t == 0:
                        sub_dict = {
                            "motion": motion_output[0].detach().cpu().numpy(),
                            "length": gen_motion_len,
                            "cap_len": cap_lens[0].item(),
                            "caption": caption[0],
                            "tokens": tokens,
                        }
                        generated_motion.append(sub_dict)
                    if is_mm:
                        mm_motions.append(
                            {
                                "motion": motion_output[0].detach().cpu().numpy(),
                                "length": gen_motion_len,
                            }
                        )

                if is_mm:
                    mm_generated_motions.append(
                        {
                            "caption": caption[0],
                            "tokens": tokens,
                            "cap_len": cap_lens[0].item(),
                            "mm_motions": mm_motions,
                        }
                    )

        self.generated_motion = generated_motion
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = (
            data["motion"],
            data["length"],
            data["caption"],
            data["tokens"],
        )
        sent_len = data["cap_len"]
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer["unk/OTHER"]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2])),
                ],
                axis=0,
            )
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens)


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.max_motion_length = motion_dataset.max_motion_length
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data["mm_motions"]
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion["length"])
            motion = mm_motion["motion"]
            if len(motion) < self.max_motion_length:
                motion = np.concatenate(
                    [
                        motion,
                        np.zeros(
                            (self.max_motion_length - len(motion), motion.shape[1], motion.shape[2])
                        ),
                    ],
                    axis=0,
                )
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=np.int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens


def get_generated_motion_loader(
    batch_size,
    vae,
    dit,
    ground_truth_dataset,
    mm_num_samples,
    mm_num_repeats,
    cond_scale,
    feature_type,
    history_length,
    sampling_timesteps,
    uncertainty_scale,
    scheduling_matrix_type,
    infer_from_velocity,
):
    start = time.time()
    dataset = EvaluationDataset(
        vae,
        dit,
        ground_truth_dataset,
        batch_size=batch_size,
        mm_num_samples=mm_num_samples,
        mm_num_repeats=mm_num_repeats,
        cond_scale=cond_scale,
        feature_type=feature_type,
        history_length=history_length,
        sampling_timesteps=sampling_timesteps,
        uncertainty_scale=uncertainty_scale,
        scheduling_matrix_type=scheduling_matrix_type,
        infer_from_velocity=infer_from_velocity,
    )
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4
    )

    if mm_dataset.dataset:
        mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)
    else:
        mm_motion_loader = None

    logger.info(f"Generated Dataset Loading Completed in {(time.time() - start) / 60:.2f} min.")

    return motion_loader, mm_motion_loader


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


def build_models(opt):
    movement_enc = MovementConvEncoder(
        opt.dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent
    )
    text_enc = TextEncoderBiGRUCo(
        word_size=opt.dim_word,
        pos_size=opt.dim_pos_ohot,
        hidden_size=opt.dim_text_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device,
    )

    motion_enc = MotionEncoderBiGRUCo(
        input_size=opt.dim_movement_latent,
        hidden_size=opt.dim_motion_hidden,
        output_size=opt.dim_coemb_hidden,
        device=opt.device,
    )

    checkpoint = torch.load(
        pjoin(opt.checkpoints_dir, opt.dataset_name, "text_mot_match", "model", "finest.tar"),
        map_location=opt.device,
    )
    movement_enc.load_state_dict(checkpoint["movement_encoder"])
    text_enc.load_state_dict(checkpoint["text_encoder"])
    motion_enc.load_state_dict(checkpoint["motion_encoder"])
    print("Loading Evaluation Model Wrapper (Epoch %d) Completed!!" % (checkpoint["epoch"]))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):
    def __init__(self, cfg):
        cfg.dim_pose = 56 * 12

        cfg.dim_word = 300
        cfg.dim_pos_ohot = len(POS_enumerator)
        cfg.dim_motion_hidden = 1024
        cfg.dim_text_hidden = 512
        cfg.dim_coemb_hidden = 512
        # interx
        cfg.max_motion_length = 150
        cfg.max_text_len = 35

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(cfg)
        self.opt = cfg
        self.device = cfg.device

        self.text_encoder.to(cfg.device)
        self.motion_encoder.to(cfg.device)
        self.movement_encoder.to(cfg.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    def prepare_motion(self, motions):
        motions[:, :, -1, 9:] = 0
        motions[:, :, -1, 3:6] = 0
        motions = motions.reshape(motions.shape[0], motions.shape[1], -1)
        return motions

    def get_co_embeddings(self, batch):
        word_embs, pos_ohot, _, cap_lens, motions, m_lens, _ = batch

        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            """Movement Encoding"""
            motions = self.prepare_motion(motions)
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            """Text Encoding"""
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    def get_motion_embeddings(self, batch):
        try:
            _, _, _, sent_lens, motions, m_lens, _ = batch
        except ValueError:
            motions, m_lens = batch
        if len(motions.shape) == 5:
            motions = motions[0]
            m_lens = m_lens[0]
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            """Movement Encoding"""
            motions = self.prepare_motion(motions)
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
