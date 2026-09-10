import numpy as np
from loguru import logger
import torch
from einops import einsum
from utils.coordinate import extract_coordinate
from utils import rigid_transform, inv_transform
from utils.ops import get_ops


def concat_features(*args, dim=-1):
    ops = get_ops(args[0])
    return ops.concat(args, dim=dim)


def interhuman_to_features(
    motion1: np.ndarray | torch.Tensor,
    motion2: np.ndarray | torch.Tensor | None = None,
    feature_type: str = "264",
    local_dim: int = 262,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    feature_type = str(feature_type)
    if feature_type == "interhuman":
        if motion2 is not None:
            return motion1, motion2
        return motion1
    ops = get_ops(motion1)

    def get_components(motion):
        R, t = extract_coordinate(motion)
        R_inv, t_inv = inv_transform(R, t, "t", "t")
        motion_local = rigid_transform(motion, R_inv, t_inv, "t", "t", "t")[:, columns_to_keep]
        sin, cos = R[..., 0, [2]], R[..., 0, [0]]
        delta_R = einsum(R[1:], R[:-1], "a b c, a d c -> a b d")
        delta_sin, delta_cos = delta_R[..., 0, [2]], delta_R[..., 0, [0]]
        delta_sin = ops.concat([ops.zeros((1, 1)), delta_sin], dim=0)
        delta_cos = ops.concat([ops.ones((1, 1)), delta_cos], dim=0)
        delta_xz = t[1:] - t[:-1]
        delta_xz = ops.concat([ops.zeros((1, 2)), delta_xz[:, [0, 2]]], dim=0)
        return R, sin, cos, delta_sin, delta_cos, t[:, [0, 2]], delta_xz, motion_local

    columns_to_keep = [1] + list(range(3, local_dim))
    R1, sin1, cos1, delta_sin1, delta_cos1, t1, delta_xz1, motion_local1 = get_components(motion1)
    relative_t1 = ops.zeros_like(t1)
    relative_sin1 = ops.zeros_like(sin1)
    relative_cos1 = ops.zeros_like(cos1)
    if motion2 is not None:
        R2, sin2, cos2, delta_sin2, delta_cos2, t2, delta_xz2, motion_local2 = get_components(
            motion2
        )
        # relative_t1 = t1 - t2
        relative_t2 = t2 - t1
        relative_R1 = einsum(R1, R2, "a b c, a d c -> a b d")
        relative_sin1, relative_cos1 = relative_R1[..., 0, [2]], relative_R1[..., 0, [0]]
        relative_R2 = einsum(R2, R1, "a b c, a d c -> a b d")
        relative_sin2, relative_cos2 = relative_R2[..., 0, [2]], relative_R2[..., 0, [0]]

    def get_features(
        sin,
        cos,
        delta_sin,
        delta_cos,
        relative_sin,
        relative_cos,
        t,
        delta_xz,
        motion_local,
        relative_t,
        semantic_idx,
    ):
        match feature_type:
            case "global_position":
                return concat_features(sin, cos, t, motion_local)
            case "264":
                if semantic_idx == 0:
                    return concat_features(sin, cos, delta_xz, motion_local)
                else:
                    return concat_features(sin, cos, relative_t, motion_local)
            case "velocity_relative_translation":
                return concat_features(sin, cos, delta_xz, relative_t, motion_local)
            case "position_velocity":
                return concat_features(sin, cos, t, delta_xz, motion_local)
            case "root":
                return concat_features(sin, cos, t, delta_xz)
            case "local":
                return motion_local[:, :-4]
            case "rotation_velocity":
                return concat_features(
                    delta_sin,
                    delta_cos,
                    relative_sin,
                    relative_cos,
                    delta_xz,
                    relative_t,
                    motion_local,
                )

    features1 = get_features(
        sin1,
        cos1,
        delta_sin1,
        delta_cos1,
        relative_sin1,
        relative_cos1,
        t1,
        delta_xz1,
        motion_local1,
        relative_t1,
        0,
    )
    if motion2 is not None:
        features2 = get_features(
            sin2,
            cos2,
            delta_sin2,
            delta_cos2,
            relative_sin2,
            relative_cos2,
            t2,
            delta_xz2,
            motion_local2,
            relative_t2,
            1,
        )
        return features1, features2
    return features1


def accumulate_rotations(dR: np.ndarray):
    """
    dR: (T, 3, 3) delta rotations
    returns R: (T, 3, 3) absolute rotations, where R[t] = R[t-1] @ dR[t]
    """
    assert dR.ndim == 3 and dR.shape[1:] == (3, 3)
    T = dR.shape[0]
    R = np.empty((T, 3, 3))
    R[0] = np.eye(3)
    for t in range(0, T - 1):
        R[t + 1] = R[t] @ dR[t + 1]
    return R


def features_to_interhuman(
    features1: torch.Tensor | np.ndarray,
    features2: torch.Tensor | np.ndarray | None = None,
    feature_type: str = "264",
    inference_from_velocity: bool = False,
    inference_from_rotation_velocity: bool = True,
    local_dim: int = 262,
) -> np.ndarray | tuple[np.ndarray, np.ndarray] | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    feature_type = str(feature_type)
    if feature_type == "interhuman":
        if features2 is not None:
            return features1, features2
        return features1
    columns_to_keep = [1] + list(range(3, local_dim))
    is_tensor = False
    if isinstance(features1, torch.Tensor):
        device = features1.device
        features1 = features1.cpu().numpy()
        is_tensor = True
        if features2 is not None:
            features2 = features2.cpu().numpy()

    def get_rotation_matrix(features1, features2=None):
        def sin_cos_to_rotation_matrix(sin, cos):
            zeros = np.zeros_like(sin)
            ones = np.ones_like(sin)
            return np.stack(
                [
                    np.stack([cos, zeros, sin], axis=-1),
                    np.stack([zeros, ones, zeros], axis=-1),
                    np.stack([-sin, zeros, cos], axis=-1),
                ],
                axis=-2,
            )

        R1 = np.zeros((features1.shape[0], 3, 3))
        R2 = R1.copy()
        match feature_type:
            case "rotation_velocity":
                d_sin1, d_cos1 = features1[..., 0], features1[..., 1]
                d_R1 = sin_cos_to_rotation_matrix(d_sin1, d_cos1)
                R1 = accumulate_rotations(d_R1)
                if features2 is not None:
                    d_sin2, d_cos2 = features2[..., 0], features2[..., 1]
                    d_R2 = sin_cos_to_rotation_matrix(d_sin2, d_cos2)
                    if inference_from_rotation_velocity:
                        R2 = accumulate_rotations(d_R2)
                        rel_sin2, rel_cos2 = features2[..., 2], features2[..., 3]
                        rel_R2 = sin_cos_to_rotation_matrix(rel_sin2, rel_cos2)
                        R2 = einsum(rel_R2[0], R2, "b c, a c d -> a b d")
                    else:
                        rel_sin2, rel_cos2 = features2[..., 2], features2[..., 3]
                        rel_R2 = sin_cos_to_rotation_matrix(rel_sin2, rel_cos2)
                        R2 = rel_R2 @ R1
            case _:
                sin1, cos1 = features1[..., 0], features1[..., 1]
                R1 = sin_cos_to_rotation_matrix(sin1, cos1)
                if features2 is not None:
                    sin2, cos2 = features2[..., 0], features2[..., 1]
                    R2 = sin_cos_to_rotation_matrix(sin2, cos2)
        return R1, R2

    def get_translation(features1, features2=None):
        t1 = np.zeros((features1.shape[0], 3))
        t2 = t1.copy()
        match feature_type:
            case "264":
                t1[:, [0, 2]] = features1[:, [2, 3]]
                t1 = t1.cumsum(axis=0)
                if features2 is not None:
                    t2[:, [0, 2]] = features2[:, [2, 3]]
                    t2 = t2 + t1
            case "velocity_relative_translation":
                t1[:, [0, 2]] = features1[:, [2, 3]]
                t1 = t1.cumsum(axis=0)
                if features2 is not None:
                    t2[:, [0, 2]] = features2[:, [4, 5]]
                    t2 = t2 + t1
                    if inference_from_velocity:
                        t2[1:, [0, 2]] = features2[1:, [2, 3]]
                        t2 = t2.cumsum(axis=0)
            case "rotation_velocity":
                t1[:, [0, 2]] = features1[:, [4, 5]]
                t1 = t1.cumsum(axis=0)
                if features2 is not None:
                    t2[:, [0, 2]] = features2[:, [6, 7]]
                    t2 = t2 + t1
                    if inference_from_velocity:
                        t2[1:, [0, 2]] = features2[1:, [4, 5]]
                        t2 = t2.cumsum(axis=0)
            case "position_velocity":
                t1[:, [0, 2]] = features1[:, [2, 3]]
                if features2 is not None:
                    t2[:, [0, 2]] = features2[:, [2, 3]]
            case "global_position":
                t1[:, [0, 2]] = features1[:, [2, 3]]
                if features2 is not None:
                    t2[:, [0, 2]] = features2[:, [2, 3]]
        return t1, t2

    def get_local_motion(features):
        local_motion = np.zeros((features.shape[0], local_dim))
        match feature_type:
            case "264":
                local_motion[:, columns_to_keep] = features[:, 4:]
            case "velocity_relative_translation":
                local_motion[:, columns_to_keep] = features[:, 6:]
            case "position_velocity":
                local_motion[:, columns_to_keep] = features[:, 6:]
            case "global_position":
                local_motion[:, columns_to_keep] = features[:, 4:]
            case "rotation_velocity":
                local_motion[:, columns_to_keep] = features[:, 8:]
        return local_motion

    R1, R2 = get_rotation_matrix(features1, features2)
    t1, t2 = get_translation(features1, features2)
    local_motion1 = get_local_motion(features1)
    global_motion1 = rigid_transform(local_motion1, R1, t1, "t", "t", "t")
    if features2 is not None:
        local_motion2 = get_local_motion(features2)
        global_motion2 = rigid_transform(local_motion2, R2, t2, "t", "t", "t")
        if is_tensor:
            global_motion1 = torch.from_numpy(global_motion1).to(device)
            global_motion2 = torch.from_numpy(global_motion2).to(device)
        return global_motion1, global_motion2
    if is_tensor:
        global_motion1 = torch.from_numpy(global_motion1).to(device)
    return global_motion1


def create_indices(
    motion_list: list[np.ndarray], window_size: int, window_stride: int
) -> list[tuple[int, int, int]]:
    indices = []
    for i, sample in enumerate(motion_list):
        if window_stride == -1:
            indices.append((i, 0, len(sample)))
        else:
            for start_idx in range(0, len(sample) - window_size + 1, window_stride):
                indices.append((i, start_idx, start_idx + window_size))
    return indices


class Normalizer:
    def __init__(self, mean_path: str | list[str], std_path: str | list[str]):
        if isinstance(mean_path, str) and isinstance(std_path, str):
            mean_path = [mean_path]
            std_path = [std_path]
            self.with_semantic_idx = False
        else:
            self.with_semantic_idx = True
        logger.info(f"Loading mean from {mean_path}")
        logger.info(f"Loading std from {std_path}")
        logger.info("with semantic idx" if self.with_semantic_idx else "without semantic idx")
        self.mean = np.stack([np.load(mean) for mean in mean_path], axis=0)
        self.std = np.stack([np.load(std) for std in std_path], axis=0)

        self.mean_torch = torch.from_numpy(self.mean)
        self.std_torch = torch.from_numpy(self.std)

    def _get_mean_std(
        self,
        data: np.ndarray | torch.Tensor,
        semantic_idx: np.ndarray | torch.Tensor | int,
    ) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
        """Get mean and std with correct type and shape for the given data.

        Handles:
        - data shapes: [B, L, D], [L, D], or [D]
        - semantic_idx: int or array/tensor of shape [B]
        - mean/std loaded shape: [num_semantic_classes, D]
        """
        dim = data.shape[-1]
        if not self.with_semantic_idx:
            semantic_idx = 0

        if isinstance(data, torch.Tensor):
            # Select from torch tensors
            if isinstance(semantic_idx, (int, np.integer)):
                # Single index: result shape [D]
                mean = self.mean_torch[semantic_idx, :dim].to(data.device).type_as(data)
                std = self.std_torch[semantic_idx, :dim].to(data.device).type_as(data)
            else:
                # Batch of indices: result shape [B, D]
                if isinstance(semantic_idx, np.ndarray):
                    semantic_idx = torch.from_numpy(semantic_idx).to(data.device)
                mean = self.mean_torch[semantic_idx, :dim].to(data.device).type_as(data)
                std = self.std_torch[semantic_idx, :dim].to(data.device).type_as(data)

                # Reshape for broadcasting: [B, D] -> [B, 1, D] if data is [B, L, D]
                if data.ndim == 3:
                    mean = mean.unsqueeze(1)
                    std = std.unsqueeze(1)
        else:
            # Select from numpy arrays
            if isinstance(semantic_idx, (int, np.integer)):
                # Single index: result shape [D]
                mean = self.mean[semantic_idx, :dim]
                std = self.std[semantic_idx, :dim]
            else:
                # Batch of indices: result shape [B, D]
                if isinstance(semantic_idx, torch.Tensor):
                    semantic_idx = semantic_idx.cpu().numpy()
                mean = self.mean[semantic_idx, :dim]
                std = self.std[semantic_idx, :dim]

                # Reshape for broadcasting: [B, D] -> [B, 1, D] if data is [B, L, D]
                if data.ndim == 3:
                    mean = np.expand_dims(mean, 1)
                    std = np.expand_dims(std, 1)

        return mean, std

    def forward(
        self, data: np.ndarray | torch.Tensor, semantic_idx: np.ndarray | torch.Tensor | int = 0
    ) -> np.ndarray | torch.Tensor:
        mean, std = self._get_mean_std(data, semantic_idx)
        return (data - mean) / std

    def backward(
        self, data: np.ndarray | torch.Tensor, semantic_idx: int = 0
    ) -> np.ndarray | torch.Tensor:
        mean, std = self._get_mean_std(data, semantic_idx)
        return data * std + mean
