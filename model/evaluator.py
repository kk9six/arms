from utils.coordinate import reset_coordinate
from loguru import logger
from dataset.utils import features_to_interhuman
import torch
from einops import repeat
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np
from model.evaluator_models import InterCLIP
from tqdm import tqdm
from model.vae import VAE
from model.dit import DiT


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
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []

        with torch.no_grad():
            for i, (text, motion1, _, motion_lens, _) in tqdm(
                enumerate(dataloader), total=len(dataloader), desc="Generating motions"
            ):
                if i in mm_idxs:
                    num_repeats = mm_num_repeats
                else:
                    num_repeats = 1

                ids_length = motion_lens.detach().long() // 4

                motion_lens = ids_length * 4

                ids_length = repeat(ids_length, "b -> b n", n=2)
                # motion1_output = torch.empty((0, motion1.shape[1], motion1.shape[2]))
                # motion2_output = torch.empty((0, motion1.shape[1], motion1.shape[2]))
                for num_repeat in range(num_repeats):
                    latent1, latent2 = dit.generate(
                        text,
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
                    output1_list, output2_list = [], []
                    for o1, o2 in zip(output1, output2):
                        o1_interhuman, o2_interhuman = features_to_interhuman(
                            o1, o2, feature_type=feature_type, inference_from_velocity=infer_from_velocity
                        )
                        o1_interhuman, o2_interhuman = reset_coordinate(
                            o1_interhuman, o2_interhuman
                        )
                        if np.random.rand() > 0.5:
                            output1_list.append(o1_interhuman)
                            output2_list.append(o2_interhuman)
                        else:
                            output1_list.append(o2_interhuman)
                            output2_list.append(o1_interhuman)
                    output1 = torch.stack(output1_list)
                    output2 = torch.stack(output2_list)
                    if num_repeat == 0:
                        motion1_output = output1
                        motion2_output = output2
                    else:
                        motion1_output = torch.cat((motion1_output, output1), dim=0)
                        motion2_output = torch.cat((motion2_output, output2), dim=0)

                # motions_output = torch.stack([motion1_output, motion2_output], dim=-2).numpy()
                # if np.random.rand() > 0.5:
                motions_output = torch.stack([motion1_output, motion2_output], dim=-2).numpy()
                # else:
                #     motions_output = torch.stack([motion2_output, motion1_output], dim=-2).numpy()
                B, T, N, D = motions_output.shape
                if T < self.max_length:
                    padding_len = self.max_length - T
                    padding_zeros = np.zeros((B, padding_len, N, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length

                for b in range(B):
                    generated_motions.append(
                        {
                            "motion1": motions_output[b, :, 0],
                            "motion2": motions_output[b, :, 1],
                            "motion_lens": motion_lens[b],
                            "text": text[b],
                        }
                    )
                    if i in mm_idxs:
                        mm_generated_motions.append(
                            {
                                "mm_motions": motions_output[
                                    b * num_repeats : (b + 1) * num_repeats
                                ],
                                "motion_lens": motion_lens[b],
                                "text": text[b],
                            }
                        )

        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = (
            data["motion1"],
            data["motion2"],
            data["motion_lens"],
            data["text"],
        )
        return text, motion1, motion2, motion_lens, motion_lens


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data["mm_motions"]
        motion_lens = data["motion_lens"]
        mm_motions1 = mm_motions[:, :, 0]
        mm_motions2 = mm_motions[:, :, 1]
        text = data["text"]
        motion_lens = np.array([motion_lens] * mm_motions1.shape[0])
        return text, mm_motions1, mm_motions2, motion_lens, motion_lens


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
        dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True
    )
    if mm_dataset.dataset:
        mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)
    else:
        mm_motion_loader = None

    logger.info(f"Generated Dataset Loading Completed in {(time.time() - start) / 60:.2f} min.")

    return motion_loader, mm_motion_loader


def build_models(cfg):
    model = InterCLIP(cfg)

    checkpoint = torch.load("checkpoints/interclip.ckpt", map_location="cpu", weights_only=False)
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


class EvaluatorModelWrapper(object):
    def __init__(self, cfg, device):
        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            text, motion1, motion2, motion_lens, motion_lens2 = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            """Motion Encoding"""
            motion_embedding = self.model.encode_motion(batch)["motion_emb"]

            """Text Encoding"""
            text_embedding = self.model.encode_text(batch)["text_emb"][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            text, motion1, motion2, motion_lens, motion_lens2 = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            """Motion Encoding"""
            motion_embedding = self.model.encode_motion(batch)["motion_emb"]

        return motion_embedding
