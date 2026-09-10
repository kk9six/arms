import torch
import random
from loguru import logger
from utils.word_vectorizer import WordVectorizer
from pathlib import Path
from torch.utils import data
from utils.rotation_conversions import rotation_6d_to_axis_angle, axis_angle_to_rotation_6d
import numpy as np
from utils.body_model import BodyModel
from tqdm import tqdm
import h5py
from dataset.utils import interhuman_to_features
from dataset.utils import create_indices
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


def interhuman_to_interx(motion1, motion2):
    if isinstance(motion1, torch.Tensor):
        device = motion1.device
        motion1 = motion1.cpu().numpy()
        motion2 = motion2.cpu().numpy()
        is_tensor = True
    else:
        is_tensor = False

    rot_6d1 = motion1[:, 22 * 6 : 22 * 6 + 55 * 6].reshape(-1, 55, 6)
    rot_6d2 = motion2[:, 22 * 6 : 22 * 6 + 55 * 6].reshape(-1, 55, 6)
    rot_6d = np.concatenate([rot_6d1, rot_6d2], axis=2)

    transl1 = motion1[:, :3].reshape(-1, 1, 3)
    transl2 = motion2[:, :3].reshape(-1, 1, 3)

    transl1_vel = np.concatenate([transl1[1:] - transl1[:-1], np.zeros((1, 1, 3))], axis=0)
    transl2_vel = np.concatenate([transl2[1:] - transl2[:-1], np.zeros((1, 1, 3))], axis=0)
    # transl1_vel = motion1[:, 22 * 3 : 22 * 3 + 3].reshape(-1, 1, 3)
    # transl2_vel = motion2[:, 22 * 3 : 22 * 3 + 3].reshape(-1, 1, 3)

    transl_vel = np.concatenate([transl1, transl1_vel, transl2, transl2_vel], axis=2)

    interx_features = np.concatenate([rot_6d, transl_vel], axis=1)
    if is_tensor:
        return torch.from_numpy(interx_features).to(device)
    return interx_features


class InterXM2M(data.Dataset):
    def __init__(
        self,
        window_size: int,
        window_stride: int,
        mode: str,
        feature_type: str = "velocity_relative_translation",
        split_features: bool = False,
        random_rotate: bool = False,
    ):
        data_root_dir = Path("dataset/InterX")
        self.data = []
        self.lengths = []
        self.window_size = window_size
        self.window_stride = window_stride
        self.mode = mode
        id_list = []
        self.motion_id_list = []
        with (data_root_dir / "splits" / f"{mode}.txt").open("r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        logger.info(f"Loading InterX dataset in {mode} mode")
        with h5py.File(data_root_dir / "processed" / "motions" / f"{mode}_processed.h5", "r") as mf:
            for name in tqdm(id_list):
                try:
                    motion = np.array(mf[name]).astype("float32")
                    if motion.shape[0] < window_size:
                        continue
                    motion1, motion2 = motion[:, 0], motion[:, 1]
                    motion1, motion2 = interhuman_to_features(
                        motion1,
                        motion2,
                        feature_type="velocity_relative_translation",
                        local_dim=466,
                    )
                    self.data.extend([motion1, motion2])
                    self.motion_id_list.extend([name, name])
                except Exception as e:
                    print(e)
        self.indices = create_indices(self.data, window_size, window_stride)
        logger.info(
            "Total number of motions {}, snippets {}".format(len(self.data), len(self.indices))
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, start_idx, end_idx = self.indices[idx]
        if sample_idx % 2 == 0:
            semantic_idx = 0
        else:
            semantic_idx = 1
        sample = self.data[sample_idx][start_idx:end_idx]
        return self.motion_id_list[sample_idx], semantic_idx, sample


class InterXT2M(data.Dataset):
    def __init__(
        self,
        mode: str,
        feature_type: str = "velocity_relative_translation",
        random_rotate: bool = False,
        max_length: int = 150,
    ):
        self.mode = mode
        self.data_root_dir = Path("dataset/InterX")
        self.motion_file = self.data_root_dir / "processed" / "motions" / f"{mode}_processed.h5"
        self.text_dir = self.data_root_dir / "processed" / "texts_processed"
        self.w_vectorizer = WordVectorizer(self.data_root_dir / "processed" / "glove", "hhi_vab")
        self.max_length = 20
        self.pointer = 0
        if self.mode == "train" or self.mode == "val":
            self.max_motion_length = 160
        else:
            self.max_motion_length = 150
        self.max_text_len = 35
        self.num_person = 2
        self.unit_length = 4
        min_motion_len = 24

        data_dict = {}
        id_list = []
        with (self.data_root_dir / "splits" / f"{mode}.txt").open("r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, "r") as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype("float32")
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    text_data = []
                    flag = False
                    with (self.text_dir / f"{name}.txt").open("r") as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split("#")
                            caption = line_split[0]
                            tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict["caption"] = caption
                            text_dict["tokens"] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                exit(-1)
                    if flag:
                        motion1, motion2 = motion[:, 0], motion[:, 1]
                        if self.mode == "train" or self.mode == "val":
                            motion1, motion2 = interhuman_to_features(
                                motion1,
                                motion2,
                                feature_type="velocity_relative_translation",
                                local_dim=466,
                            )
                            motion = np.stack([motion1, motion2], axis=1)
                        else:
                            motion = interhuman_to_interx(motion1, motion2)
                        data_dict[name] = {
                            "motion": motion,
                            "length": len(motion),
                            "text": text_data,
                        }
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e:
                    print(e)
                    pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
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
        coin2 = np.random.choice(["single", "single", "double"])
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]
        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2])),
                ],
                axis=0,
            )
        else:
            motion = motion[: self.max_motion_length]
            m_length = self.max_motion_length

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens)


class InterXT2MNative(data.Dataset):
    def __init__(
        self,
        mode: str,
        feature_type: str = "velocity_relative_translation",
        random_rotate: bool = False,
        max_length: int = 150,
    ):
        self.mode = mode
        self.data_root_dir = Path("dataset/InterX")
        self.motion_file = self.data_root_dir / "processed" / "motions" / f"{mode}.h5"
        self.text_dir = self.data_root_dir / "processed" / "texts_processed"
        self.w_vectorizer = WordVectorizer(self.data_root_dir / "processed" / "glove", "hhi_vab")
        self.max_length = 20
        self.pointer = 0
        if self.mode == "train" or self.mode == "val":
            self.max_motion_length = 160
        else:
            self.max_motion_length = 150
        self.max_text_len = 35
        self.num_person = 2
        self.unit_length = 4
        min_motion_len = 24

        data_dict = {}
        id_list = []
        with (self.data_root_dir / "splits" / f"{mode}.txt").open("r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        new_name_list = []
        length_list = []
        with h5py.File(self.motion_file, "r") as mf:
            self.keys = list(mf.keys())
            for name in tqdm(id_list):
                try:
                    motion = mf[name][:].astype("float32")
                    if (len(motion)) < min_motion_len or (len(motion) >= 1000):
                        continue
                    text_data = []
                    flag = False
                    with (self.text_dir / f"{name}.txt").open("r") as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split("#")
                            caption = line_split[0]
                            tokens = line_split[1].split(" ")
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict["caption"] = caption
                            text_dict["tokens"] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                exit(-1)
                    if flag:
                        data_dict[name] = {
                            "motion": motion,
                            "length": len(motion),
                            "text": text_data,
                        }
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e:
                    print(e)
                    pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[: self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
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
        coin2 = np.random.choice(["single", "single", "double"])
        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx : idx + m_length]

        rot_6d1 = axis_angle_to_rotation_6d(
            torch.from_numpy(motion[:, :-1, :3]).float(), interhuman=False
        ).numpy()
        rot_6d2 = axis_angle_to_rotation_6d(
            torch.from_numpy(motion[:, :-1, 3:]).float(), interhuman=False
        ).numpy()
        rot_6d = np.concatenate([rot_6d1, rot_6d2], axis=-1)
        transl = motion[:, -1, :]
        vel = transl[1:] - transl[:-1]
        vel = np.concatenate([vel, np.zeros((1, vel.shape[-1]))], axis=0)
        transl_vel_all = []
        for ii in range(self.num_person):
            transl_vel_all.append(
                np.concatenate(
                    [transl[:, 3 * ii : 3 * ii + 3], vel[:, 3 * ii : 3 * ii + 3]], axis=1
                )
            )
        transl_vel_all = np.concatenate(transl_vel_all, axis=1)

        motion = np.concatenate([rot_6d, transl_vel_all.reshape(-1, 1, 12)], axis=1)

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2])),
                ],
                axis=0,
            )
        else:
            motion = motion[: self.max_motion_length]
            m_length = self.max_motion_length

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, "_".join(tokens)


if __name__ == "__main__":

    def process_motion(motion, feet_thre, bm):
        foot_id_left = [7, 10]
        foot_id_right = [8, 11]
        poses, trans = motion[:, :55], motion[:, -1]
        poses_6d = axis_angle_to_rotation_6d(poses)
        res = bm(full_pose=poses, trans=trans)
        joints = res.Jtr[:, :22]
        # put foot to ground
        joints[:, :, 1] -= joints.reshape(-1, 3).min(dim=0).values[1]
        velocities = joints[1:] - joints[:-1]

        def foot_detect(positions, thres):
            velfactor, heightfactor = torch.tensor([thres, thres]), torch.tensor([0.12, 0.05])

            feet_l_x = (positions[1:, foot_id_left, 0] - positions[:-1, foot_id_left, 0]) ** 2
            feet_l_y = (positions[1:, foot_id_left, 1] - positions[:-1, foot_id_left, 1]) ** 2
            feet_l_z = (positions[1:, foot_id_left, 2] - positions[:-1, foot_id_left, 2]) ** 2
            feet_l_h = positions[:-1, foot_id_left, 1]
            feet_l = (
                ((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)
            ).float()

            feet_r_x = (positions[1:, foot_id_right, 0] - positions[:-1, foot_id_right, 0]) ** 2
            feet_r_y = (positions[1:, foot_id_right, 1] - positions[:-1, foot_id_right, 1]) ** 2
            feet_r_z = (positions[1:, foot_id_right, 2] - positions[:-1, foot_id_right, 2]) ** 2
            feet_r_h = positions[:-1, foot_id_right, 1]
            feet_r = (
                ((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)
            ).float()
            return feet_l, feet_r

        feet_l, feet_r = foot_detect(joints, feet_thre)
        data = torch.cat(
            [
                joints[:-1].reshape(-1, 22 * 3),
                velocities.reshape(-1, 22 * 3),
                poses_6d[:-1].reshape(-1, 55 * 6),
                feet_l,
                feet_r,
            ],
            dim=-1,
        )
        return data

    data_root_dir = "dataset/InterX"
    neutral_bm_path = "dataset/smpl/smplx/SMPLX_NEUTRAL.npz"
    bm = BodyModel(bm_fname=neutral_bm_path, num_betas=10)
    for mode in ["train", "val", "test"]:
        print(f"Processing {mode}...")
        output_file = h5py.File(f"{data_root_dir}/processed/motions/{mode}_processed.h5", "w")
        h5_file = h5py.File(f"{data_root_dir}/processed/motions/{mode}.h5", "r")
        for key in tqdm(h5_file.keys()):
            motion = torch.tensor(np.array(h5_file[key]))
            motion1, motion2 = motion[..., :3].clone(), motion[..., 3:].clone()
            motion1_final = process_motion(motion1, 0.001, bm)
            motion2_final = process_motion(motion2, 0.001, bm)
            motion_final = torch.stack([motion1_final, motion2_final], dim=1).detach().cpu().numpy()
            output_file.create_dataset(key, data=motion_final, dtype="f4")
