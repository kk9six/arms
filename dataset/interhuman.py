import numpy as np
from utils.coordinate import recover_coordinate_features
import random
from typing import Literal
from pathlib import Path
from torch.utils import data
from loguru import logger
from tqdm import tqdm
from utils import rigid_transform
from dataset.utils import create_indices, interhuman_to_features


class InterHumanM2M(data.Dataset):
    def __init__(
        self,
        mode: str = "train",
        feature_type: str = "264",
        window_size: int = 64,
        window_stride: int = 10,
        split_features: bool = False,
        random_rotate: bool = False,
    ) -> None:
        self.feature_type = feature_type
        self.split_features = split_features

        if split_features:
            self.rel_columns = [4, 5]
            self.main_columns = np.setdiff1d(list(range(266)), self.rel_columns).tolist()
            assert feature_type == "velocity_relative_translation", (
                "split_features is only supported for velocity_relative_translation feature type"
            )

        data_root_dir = Path("dataset/InterHuman")
        logger.info(f"Loading InterHuman dataset in {mode} mode")
        min_length = window_size + 1  # minimum length of a motion is window_size + 1

        id_list = (data_root_dir / "split" / f"{mode}.txt").read_text().splitlines()
        self.motion_list = []
        self.motion_id_list = []
        for id in tqdm(id_list, desc="Loading motions"):
            fpath_person1 = data_root_dir / "motions_processed" / "person1" / f"{id}.npy"
            motion1, motion1_swap = load_motion(fpath_person1, min_length, swap=True)
            if motion1 is None:
                continue
            fpath_person2 = data_root_dir / "motions_processed" / "person2" / f"{id}.npy"
            motion2, motion2_swap = load_motion(fpath_person2, min_length, swap=True)
            if motion2 is None:
                continue

            def process_both_motions(motion1, motion2):
                """
                motion 1 and motion 2 are in the seperate global coordnate systems.
                motion1_rel is in the global coordinate system of motion 2.
                motion2_rel is in the global coordinate system of motion 1.
                """
                motion1, R1, t1 = process_motion(motion1, 0.001)
                motion2, R2, t2 = process_motion(motion2, 0.001)

                motion2_rel = rigid_transform(motion2, R1.T @ R2, R1.T @ (t2 - t1), "t", "", "")
                motion1_rel = rigid_transform(motion1, R2.T @ R1, R2.T @ (t1 - t2), "t", "", "")
                return motion1, motion2, motion1_rel, motion2_rel

            m1, m2, m1_rel, m2_rel = process_both_motions(motion1, motion2)
            m1_swap, m2_swap, m1_swap_rel, m2_swap_rel = process_both_motions(
                motion1_swap, motion2_swap
            )
            # pairs = [(m1, m2_rel), (m2, m1_rel), (m1_swap, m2_swap_rel), (m2_swap, m1_swap_rel)]
            pairs = [(m1, m2_rel), (m1_swap, m2_swap_rel)]
            for p1, p2 in pairs:
                p1, p2 = interhuman_to_features(p1, p2, feature_type=self.feature_type)
                self.motion_list.extend([p1, p2])
                self.motion_id_list.extend([id, id])
        self.indices = create_indices(self.motion_list, window_size, window_stride)
        logger.info(f"Total number of motions {len(self.motion_list)}, samples {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[str, int, np.ndarray]:
        sample_idx, start_idx, end_idx = self.indices[idx]
        if sample_idx % 2 == 0:
            semantic_idx = 0
        else:
            semantic_idx = 1
        sample = self.motion_list[sample_idx][start_idx:end_idx].copy()



        if self.split_features:
            main_features = sample[:, self.main_columns]
            rel_features = sample[:, self.rel_columns]
            # motion_id, semantic_idx, main_features, rel_features, mask_rel
            return self.motion_id_list[sample_idx], 0, 1, main_features, rel_features
        return self.motion_id_list[sample_idx], semantic_idx, sample


class InterHumanT2M(data.Dataset):
    def __init__(
        self,
        data_root_dir: str | Path = "dataset/InterHuman",
        mode: Literal["train", "val", "test"] = "train",
        min_length: int = 15,
        max_length: int = 300,
        feature_type: str = "264",
        random_rotate: bool = False,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.feature_type = feature_type
        self.data = []
        self.mode = mode
        self.random_rotate = random_rotate
        data_root_dir = Path(data_root_dir)
        id_list = (data_root_dir / "split" / f"{mode}.txt").read_text().splitlines()
        logger.info(f"Loading InterHuman Text to Motion dataset in {mode} mode")
        logger.info(f"Min length: {self.min_length}, Max length: {self.max_length}")
        for id in tqdm(id_list, desc="Loading motions"):
            fpath_person1 = data_root_dir / "motions_processed" / "person1" / f"{id}.npy"
            motion1, motion1_swap = load_motion(fpath_person1, self.min_length, swap=True)
            if motion1 is None:
                continue
            fpath_person2 = data_root_dir / "motions_processed" / "person2" / f"{id}.npy"
            motion2, motion2_swap = load_motion(fpath_person2, self.min_length, swap=True)
            text_fpath = data_root_dir / "annots" / f"{id}.txt"
            try:
                texts = [item.strip() for item in text_fpath.read_text().splitlines()]
            except Exception as e:
                logger.error(f"Reading text file failed: {text_fpath} ({e})")
                continue
            self.data.append(
                {
                    "motion1": motion1,
                    "motion2": motion2,
                    "texts": texts,
                }
            )
            if mode == "train":
                texts_swap = [
                    item.replace("left", "tmp")
                    .replace("right", "left")
                    .replace("tmp", "right")
                    .replace("clockwise", "tmp")
                    .replace("counterclockwise", "clockwise")
                    .replace("tmp", "counterclockwise")
                    for item in texts
                ]
                self.data.append(
                    {
                        "motion1": motion1_swap,
                        "motion2": motion2_swap,
                        "texts": texts_swap,
                    }
                )
        logger.info(f"Loaded {len(self.data)} motions")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        text = random.choice(self.data[idx]["texts"])

        motion1, motion2 = self.data[idx]["motion1"], self.data[idx]["motion2"]
        length = motion1.shape[0]
        start_idx = 0
        if length > self.max_length:
            start_idx = np.random.randint(0, length - self.max_length + 1)
        motion1 = motion1[start_idx : start_idx + self.max_length + 1]
        motion2 = motion2[start_idx : start_idx + self.max_length + 1]

        if np.random.rand() > 0.5:
            motion1, motion2 = motion2, motion1

        motion1, R1, t1 = process_motion(motion1, 0.001)
        motion2, R2, t2 = process_motion(motion2, 0.001)
        motion2 = rigid_transform(motion2, R1.T @ R2, R1.T @ (t2 - t1), "t", "", "")

        if self.mode == "train" or self.mode == "val":
            motion1, motion2 = interhuman_to_features(
                motion1, motion2, feature_type=self.feature_type
            )
            if self.random_rotate:
                yaw = np.random.uniform(-np.pi, np.pi, size=(1,))
                sin, cos = np.sin(yaw), np.cos(yaw)
                motion1 = recover_coordinate_features(motion1, (sin, cos, 0, 0, 0, 0))
                motion2 = recover_coordinate_features(motion2, (sin, cos, 0, 0, 0, 0))

        else:
            if np.random.rand() > 0.5:
                motion1, motion2 = motion2, motion1

        length1 = motion1.shape[0]
        length2 = motion2.shape[0]
        if length1 < self.max_length:
            paddings = np.zeros((self.max_length - length1, motion1.shape[1]))
            motion1 = np.concatenate([motion1, paddings], axis=0)
            paddings = np.zeros((self.max_length - length2, motion2.shape[1]))
            motion2 = np.concatenate([motion2, paddings], axis=0)
        length1 = length1 // 4 * 4
        length2 = length2 // 4 * 4

        assert motion1.shape[0] == self.max_length
        assert motion2.shape[0] == self.max_length

        return text, motion1, motion2, length1, length2


def load_motion(file_path, min_length, swap=False):
    try:
        motion = np.load(file_path).astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading motion from {file_path}: {e}")
        return None, None
    motion1 = motion[:, : 22 * 3]
    motion2 = motion[:, 62 * 3 : 62 * 3 + 21 * 6]
    motion = np.concatenate([motion1, motion2], axis=1)

    if motion.shape[0] < min_length:
        return None, None
    if swap:
        motion_swap = swap_left_right(motion, 22)
    else:
        motion_swap = None
    return motion, motion_swap


def swap_left_right_position(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [
        22,
        23,
        24,
        34,
        35,
        36,
        25,
        26,
        27,
        31,
        32,
        33,
        28,
        29,
        30,
        52,
        53,
        54,
        55,
        56,
    ]
    right_hand_chain = [
        43,
        44,
        45,
        46,
        47,
        48,
        40,
        41,
        42,
        37,
        38,
        39,
        49,
        50,
        51,
        57,
        58,
        59,
        60,
        61,
    ]

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right_rot(data):
    assert len(data.shape) == 3 and data.shape[-1] == 6
    data = data.copy()

    data[..., [1, 2, 4]] *= -1

    right_chain = np.array([2, 5, 8, 11, 14, 17, 19, 21]) - 1
    left_chain = np.array([1, 4, 7, 10, 13, 16, 18, 20]) - 1
    left_hand_chain = (
        np.array(
            [
                22,
                23,
                24,
                34,
                35,
                36,
                25,
                26,
                27,
                31,
                32,
                33,
                28,
                29,
                30,
            ]
        )
        - 1
    )
    right_hand_chain = (
        np.array(
            [
                43,
                44,
                45,
                46,
                47,
                48,
                40,
                41,
                42,
                37,
                38,
                39,
                49,
                50,
                51,
            ]
        )
        - 1
    )

    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


def swap_left_right(data, n_joints):
    T = data.shape[0]
    new_data = data.copy()
    positions = new_data[..., : 3 * n_joints].reshape(T, n_joints, 3)
    rotations = new_data[..., 3 * n_joints :].reshape(T, -1, 6)

    positions = swap_left_right_position(positions)
    rotations = swap_left_right_rot(rotations)

    new_data = np.concatenate([positions.reshape(T, -1), rotations.reshape(T, -1)], axis=-1)
    return new_data


def process_motion(
    motion: np.ndarray, feet_thre: float = 0.001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    rhip_id, lhip_id = 2, 1

    positions = motion[:, : 22 * 3].reshape(-1, 22, 3)
    rotations = motion[:, 22 * 3 : 22 * 3 + 21 * 6]

    positions = np.einsum("mn, tjn->tjm", trans_matrix, positions)
    positions[:, :, 1] -= positions.reshape(-1, 3).min(axis=0)[1]

    def extract_coordinate() -> tuple[np.ndarray, np.ndarray]:
        # positions: [T, J, 3]
        x_axis = positions[0, lhip_id] - positions[0, rhip_id]
        x_axis[1] = 0.0
        x_axis = x_axis / np.linalg.norm(x_axis, axis=-1)
        y_axis = np.array([0, 1, 0])
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis, axis=-1)
        R = np.stack([x_axis, y_axis, z_axis], axis=-1)
        t = positions[0, 0] * np.array([1.0, 0.0, 1.0])
        return R, t

    R, t = extract_coordinate()
    positions = np.einsum("ij,ntj->nti", R.transpose(-1, -2), (positions - t.reshape(1, 1, 3)))

    def foot_detect():
        velfactor, heightfactor = np.array([feet_thre, feet_thre]), np.array([0.12, 0.05])
        foot_id_left = [7, 10]
        foot_id_right = [8, 11]

        feet_l = ((positions[1:, foot_id_left] - positions[:-1, foot_id_left]) ** 2).sum(axis=-1)
        feet_l_h = positions[:-1, foot_id_left, 1]
        feet_l = ((feet_l < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r = ((positions[1:, foot_id_right] - positions[:-1, foot_id_right]) ** 2).sum(axis=-1)
        feet_r_h = positions[:-1, foot_id_right, 1]
        feet_r = ((feet_r < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect()

    joint_positions = positions.reshape(len(positions), -1)
    joint_vels = positions[1:] - positions[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    return (
        np.concatenate([joint_positions[:-1], joint_vels, rotations[:-1], feet_l, feet_r], axis=-1),
        R,
        t,
    )


if __name__ == "__main__":
    import pickle
    import torch
    from utils.body_model import BodyModel
    from utils.rotation_conversions import axis_angle_to_rotation_6d

    data_root_dir = Path("dataset/InterHuman")
    motion_dir = data_root_dir / "motions"
    output_dir = data_root_dir / "motions_processed without shape"

    neutral_bm_path = "dataset/smpl/smplh/neutral/model.npz"
    num_betas = 10
    bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas)

    fpath_list = list(motion_dir.glob("*.pkl"))
    for fpath in tqdm(fpath_list):
        with fpath.open("rb") as f:
            motion = pickle.load(f)
            motion_id = fpath.stem

            for label in ["person1", "person2"]:
                trans = torch.from_numpy(motion[label]["trans"]).float()[::2]
                root_orient = torch.from_numpy(motion[label]["root_orient"]).float()[::2]
                pose_body = (
                    torch.from_numpy(motion[label]["pose_body"])
                    .float()
                    .reshape(-1, 21, 3)[::2]
                )
                joints = bm(
                    pose_body=pose_body,
                    pose_root=root_orient,
                    trans=trans,
                    root2zero=False,
                ).Jtr[:, :22].numpy().reshape(-1, 22 * 3)
                rotations = (
                    axis_angle_to_rotation_6d(pose_body, interhuman=True)
                    .numpy()
                    .reshape(-1, 21 * 6)
                )

                features = np.zeros((joints.shape[0], 312))
                features[:, : 22 * 3] = joints
                features[:, 62 * 3 : 62 * 3 + 21 * 6] = rotations

                person_output_dir = output_dir / label
                person_output_dir.mkdir(parents=True, exist_ok=True)
                np.save(person_output_dir / f"{motion_id}.npy", features)
