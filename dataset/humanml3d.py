from tqdm import tqdm
from typing import Literal
import numpy as np
import random
from utils.coordinate import reset_coordinate
from pathlib import Path
from torch.utils import data
from loguru import logger
from dataset.utils import create_indices, interhuman_to_features
from utils.rotation_conversions import yaw_to_matrix
from utils import rigid_transform


class HumanML3DM2M(data.Dataset):
    def __init__(
        self,
        window_size: int = 64,
        window_stride: int = 10,
        mode: str = "train",
        feature_type: str = "264",
        split_features: bool = False,
        random_rotate: bool = False,
    ):
        self.split_features = split_features

        if split_features:
            self.rel_columns = [4, 5]
            self.main_columns = np.setdiff1d(list(range(266)), self.rel_columns).tolist()
            assert feature_type == "velocity_relative_translation", (
                "split_features is only supported for velocity_relative_translation feature type"
            )
        data_root_dir = Path("dataset/humanml3d_272")
        logger.info(f"Loading HumanML3D dataset in {mode} mode")
        id_list = (data_root_dir / "split" / f"{mode}.txt").read_text().splitlines()
        self.motion_list = []
        self.motion_id_list = []
        self.feature_type = feature_type
        min_length = window_size
        for id in tqdm(id_list, desc="Loading motions"):
            fpath = data_root_dir / "motion_processed" / f"{id.strip()}.npy"
            if not fpath.exists():
                continue
            motion = np.load(fpath)
            if motion.shape[0] < min_length:
                continue
            self.motion_list.append(
                interhuman_to_features(motion, feature_type=self.feature_type)
            )
            self.motion_id_list.append(id.strip())

        self.indices = create_indices(self.motion_list, window_size, window_stride)
        logger.info(f"Total number of motions {len(self.motion_list)}, samples {len(self.indices)}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[str, int, np.ndarray]:
        motion_idx, start_idx, end_idx = self.indices[idx]
        motion = self.motion_list[motion_idx]
        item = motion[start_idx:end_idx].copy()

        if np.random.rand() > 0.5:
            item[:, [4, 5]] = np.random.uniform(-5, 5, size=(item.shape[0], 2))
            item[1:, [4, 5]] = np.cumsum(item[1:, [2, 3]], axis=0)
            item[:, [4, 5]] = item[:, [4, 5]] + np.random.normal(0, 0.06, size=(item.shape[0], 2))

        if self.split_features:
            main_features = item[:, self.main_columns]
            rel_features = item[:, self.rel_columns]
            return self.motion_id_list[motion_idx], 0, 2, main_features, rel_features
        return self.motion_id_list[motion_idx], 0, item


class HumanML3DT2M(data.Dataset):
    def __init__(
        self,
        mode: str = "train",
        max_length: int = 300,
        feature_type: str = "264",
        random_rotate: bool = False,
    ):
        data_root_dir = Path("dataset/humanml3d_272")
        logger.info(f"Loading HumanML3D Text to Motion dataset in {mode} mode")
        self.min_length = 15
        self.max_length = max_length
        logger.info(f"Max length: {self.max_length}")
        self.data = []
        self.feature_type = feature_type
        self.mode = mode
        self.random_rotate = random_rotate
        id_list = (data_root_dir / "split" / f"{mode}.txt").read_text().splitlines()
        for id in tqdm(id_list, desc="Loading motions"):
            fpath = data_root_dir / "motion_processed" / f"{id.strip()}.npy"
            text_fpath = data_root_dir / "texts" / f"{id}.txt"
            if not fpath.exists() or not text_fpath.exists():
                continue
            motion = np.load(fpath)
            if motion.shape[0] < self.min_length:
                continue
            texts = [item.strip().split("#")[0] for item in text_fpath.read_text().splitlines()]
            self.data.append(
                {
                    "motion": motion,
                    "texts": texts,
                    "id": id.strip(),
                }
            )

        logger.info(f"Total number of motions {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        text = random.choice(self.data[idx]["texts"])
        motion = self.data[idx]["motion"]
        length = motion.shape[0]
        start_idx = 0
        if length > self.max_length:
            start_idx = np.random.randint(0, length - self.max_length + 1)
        motion = motion[start_idx : start_idx + self.max_length]
        motion = reset_coordinate(motion)

        if self.random_rotate:
            yaw = np.random.uniform(-np.pi, np.pi, size=(1,))
            R = yaw_to_matrix(yaw).squeeze(0)
            t = np.zeros((3,))
            motion = rigid_transform(motion, R, t, "t", "", "")

        if self.mode == "train" or self.mode == "val":
            motion = interhuman_to_features(motion, feature_type=self.feature_type)

        length = motion.shape[0]
        if length < self.max_length:
            paddings = np.zeros((self.max_length - length, motion.shape[1]))
            motion = np.concatenate([motion, paddings], axis=0)
        length = length // 4 * 4

        assert motion.shape[0] == self.max_length

        return text, motion, np.zeros_like(motion), length, 0
        # return text, motion, length


if __name__ == "__main__":
    import json
    from tqdm import tqdm
    import torch
    from utils.rotation_conversions import (
        axis_angle_to_rotation_6d,
        rotation_6d_to_matrix,
        matrix_to_axis_angle,
    )
    from utils.body_model import BodyModel

    num_betas = 10
    male_bm_path = "dataset/smpl/smplh/male/model.npz"
    female_bm_path = "dataset/smpl/smplh/female/model.npz"
    neutral_bm_path = "dataset/smpl/smplh/neutral/model.npz"
    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas)
    female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas)
    neutral_bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas)

    def accumulate_rotations(relative_rotations):
        R_total = [relative_rotations[0]]
        for R_rel in relative_rotations[1:]:
            R_total.append(np.matmul(R_rel, R_total[-1]))

        return np.array(R_total)

    def recover_from_local_rotation(final_x, njoint):
        nfrm, _ = final_x.shape
        rotations_matrix = rotation_6d_to_matrix(
            torch.from_numpy(final_x[:, 8 + 6 * njoint : 8 + 12 * njoint]).reshape(nfrm, -1, 6)
        ).numpy()
        global_heading_diff_rot = final_x[:, 2:8]
        velocities_root_xy_no_heading = final_x[:, :2]
        positions_no_heading = final_x[:, 8 : 8 + 3 * njoint].reshape(nfrm, -1, 3)
        height = positions_no_heading[:, 0, 1]

        global_heading_rot = accumulate_rotations(
            rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
        )
        inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
        rotations_matrix[:, 0, ...] = np.matmul(inv_global_heading_rot, rotations_matrix[:, 0, ...])
        velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
        velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
        velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
        velocities_root_xyz_no_heading[1:, :] = np.matmul(
            inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :, None]
        ).squeeze(-1)
        root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
        root_translation[:, 1] = height
        smpl_85 = rotations_matrix_to_smpl85(rotations_matrix, root_translation)
        return smpl_85

    def rotations_matrix_to_smpl85(rotations_matrix, translation):
        nfrm, njoint, _, _ = rotations_matrix.shape
        axis_angle = (
            matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
        )
        smpl_85 = np.concatenate(
            [axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1
        )
        return smpl_85

    amass_dir = Path.home() / Path("datasets/AMASS")
    motion_dir = Path.home() / Path("datasets/humanml3d_272/motion_data")
    motion_processed_with_shape_dir = Path("dataset/humanml3d_272/motion_processed")
    motion_processed_without_shape_dir = Path(
        "dataset/humanml3d_272/motion_processed without shape"
    )
    motion_processed_with_shape_dir.mkdir(parents=True, exist_ok=True)
    motion_processed_without_shape_dir.mkdir(parents=True, exist_ok=True)
    motion_id_to_fpath = json.load(Path("dataset/humanml3d_272/humanml3d.json").open())

    motion_dir_list = list(motion_dir.glob("*.npy"))
    for i, motion_fpath in enumerate(tqdm(motion_dir_list)):
        motion_id = motion_fpath.stem
        if motion_id.startswith("M"):
            motion_raw_id = motion_id[1:]
        else:
            motion_raw_id = motion_id
        motion = np.load(motion_dir / f"{motion_id}.npy")
        smpl_85 = recover_from_local_rotation(motion, 22)
        pose_root = smpl_85[:, :3]
        pose_body = smpl_85[:, 3:66]
        translation = smpl_85[:, 72:75]
        raw_motion = np.load(amass_dir / f"{motion_id_to_fpath[motion_raw_id]['path']}.npz")

        gender = raw_motion["gender"]
        betas = torch.from_numpy(raw_motion["betas"][:10]).float()

        if gender == "male":
            gendered_bm = male_bm
        else:
            gendered_bm = female_bm

        processing_variants = [
            (motion_processed_with_shape_dir, gendered_bm, betas),
            (motion_processed_without_shape_dir, neutral_bm, None),
        ]
        for motion_processed_dir, bm, shape_betas in processing_variants:
            positions = (
                bm(
                    pose_root=torch.from_numpy(pose_root).float(),
                    pose_body=torch.from_numpy(pose_body.reshape(-1, 21, 3)).float(),
                    trans=torch.from_numpy(translation).float(),
                    betas=shape_betas,
                )
                .Jtr[:, :22]
                .numpy()
            )
            positions[:, :, 1] -= positions.reshape(-1, 3).min(axis=0)[1]

            def extract_coordinate() -> tuple[np.ndarray, np.ndarray]:
                lhip_id, rhip_id = 1, 2
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
            positions = np.einsum(
                "ij,ntj->nti",
                R.transpose(-1, -2),
                positions - t.reshape(1, 1, 3),
            )

            rotations = (
                axis_angle_to_rotation_6d(
                    torch.from_numpy(pose_body).float().reshape(-1, 21, 3),
                    interhuman=True,
                )
                .numpy()
                .reshape(-1, 21 * 6)
            )

            def foot_detect():
                feet_thre = 0.001
                velfactor = np.array([feet_thre, feet_thre])
                heightfactor = np.array([0.12, 0.05])
                foot_id_left = [7, 10]
                foot_id_right = [8, 11]

                feet_l = (
                    (positions[1:, foot_id_left] - positions[:-1, foot_id_left]) ** 2
                ).sum(axis=-1)
                feet_l_h = positions[:-1, foot_id_left, 1]
                feet_l = ((feet_l < velfactor) & (feet_l_h < heightfactor)).astype(
                    np.float32
                )

                feet_r = (
                    (positions[1:, foot_id_right] - positions[:-1, foot_id_right]) ** 2
                ).sum(axis=-1)
                feet_r_h = positions[:-1, foot_id_right, 1]
                feet_r = ((feet_r < velfactor) & (feet_r_h < heightfactor)).astype(
                    np.float32
                )
                return feet_l, feet_r

            feet_l, feet_r = foot_detect()
            joint_vels = positions[1:] - positions[:-1]

            joint_positions = positions.reshape(len(positions), -1)
            joint_vels = joint_vels.reshape(len(joint_vels), -1)

            motion_processed = np.concatenate(
                [joint_positions[:-1], joint_vels, rotations[:-1], feet_l, feet_r],
                axis=-1,
            )
            if i == 0:
                print(motion_processed_dir, motion_processed.shape)
            np.save(motion_processed_dir / f"{motion_id}.npy", motion_processed)
