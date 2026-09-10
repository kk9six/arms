import torch
from utils import inv_transform, rigid_transform, matvec
import numpy as np
from einops import rearrange, repeat


def extract_coordinate(
    motion: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    if isinstance(motion, np.ndarray):

        def stack_fn(x):
            return np.stack(x, axis=-1)

        def dim_fn(x):
            return x.ndim

        y_axis = np.array([0, 1, 0], dtype=motion.dtype)
        mask_y = np.array([1, 0, 1], dtype=motion.dtype)

        def norm_fn(x):
            return np.clip(np.linalg.norm(x, axis=-1, keepdims=True), a_min=1e-8, a_max=None)

        def cross_fn(x, y):
            return np.cross(x, y, axis=-1)
    else:

        def stack_fn(x):
            return torch.stack(x, dim=-1)

        def dim_fn(x):
            return x.dim()

        y_axis = torch.tensor([0, 1, 0], dtype=motion.dtype, device=motion.device)
        mask_y = torch.tensor([1, 0, 1], dtype=motion.dtype, device=motion.device)

        def norm_fn(x):
            return torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=1e-12)

        def cross_fn(x, y):
            return torch.cross(x, y, dim=-1)

    if dim_fn(motion) == 3:
        B, T, _ = motion.shape
        output_shape = (
            B,
            T,
        )
    elif dim_fn(motion) == 2:
        T, _ = motion.shape
        motion = rearrange(motion, "n d -> 1 n d")
        B = 1
        output_shape = (T,)
    elif dim_fn(motion) == 1:
        motion = rearrange(motion, "d -> 1 1 d")
        B, T = 1, 1
        output_shape = ()
    else:
        raise ValueError(f"Invalid motion shape: {motion.shape}")

    lhip_id, rhip_id = 1, 2
    joints = motion[..., : 22 * 3].reshape(B, T, 22, 3)
    x_axis = joints[:, :, lhip_id] - joints[:, :, rhip_id]
    x_axis = x_axis * repeat(mask_y, "d -> b t d", b=B, t=T)
    x_axis = x_axis / norm_fn(x_axis)
    y_axis = repeat(y_axis, "d -> b t d", b=B, t=T)
    z_axis = cross_fn(x_axis, y_axis)
    z_axis = z_axis / norm_fn(z_axis)
    R = stack_fn([x_axis, y_axis, z_axis])
    t = joints[:, :, 0]
    t = t * repeat(mask_y, "d -> b t d", b=B, t=T)
    return R.reshape(output_shape + (3, 3)), t.reshape(output_shape + (3,))


def reset_coordinate(
    motion: np.ndarray | torch.Tensor,
    motion2: np.ndarray | torch.Tensor | None = None,
    reference: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    shape = motion.shape
    if len(shape) == 3:  # batch
        motion_shape = "b t"
        R_shape = "b"
        t_shape = "b"
    elif len(shape) == 2:  # sequence
        motion_shape = "t"
        R_shape = ""
        t_shape = ""

    if reference is None:
        if len(shape) == 3:
            R, t = extract_coordinate(motion[:, 0])
        else:
            R, t = extract_coordinate(motion[0])
        R_inv, t_inv = inv_transform(R, t, R_shape, t_shape)
        if motion2 is None:
            return rigid_transform(motion, R_inv, t_inv, motion_shape, R_shape, t_shape)
        else:
            return rigid_transform(
                motion, R_inv, t_inv, motion_shape, R_shape, t_shape
            ), rigid_transform(motion2, R_inv, t_inv, motion_shape, R_shape, t_shape)
    else:
        if len(shape) == 3:
            R, t = extract_coordinate(reference[:, 0])
        else:
            R, t = extract_coordinate(reference[0])
        return rigid_transform(motion, R, t, motion_shape, R_shape, t_shape)


def extract_transformation(motion1: np.ndarray | torch.Tensor, motion2: np.ndarray | torch.Tensor):
    """
    both motion1 and motion2 are in the motion1's coordinate system, that means, motion1 is in the zero coordinate

    motion_2 = rigid_transform(motion2_origin, R_1_to_2, t_1_to_2, "t", "", "")
    """
    R1, t1 = extract_coordinate(motion1)
    R2, t2 = extract_coordinate(motion2)
    if isinstance(R1, np.ndarray):
        R_1_to_2 = np.matmul(R1.transpose(0, 2, 1), R2)
        t_1_to_2 = matvec(R1.transpose(0, 2, 1), (t2 - t1), "b", "b")
    else:
        R_1_to_2 = torch.matmul(R1.transpose(2, 1), R2)
        t_1_to_2 = matvec(R1.transpose(2, 1), (t2 - t1), "t", "t")
    return R_1_to_2[0], t_1_to_2[0]


def reset_coordinate_features(features: torch.Tensor, transformation=None):
    features = features.clone()
    sin, cos = features[..., 0].clone(), features[..., 1].clone()
    if transformation is None:
        sin_0, cos_0 = sin[..., 0].clone(), cos[..., 0].clone()
    else:
        sin_0, cos_0 = transformation[0], transformation[1]
    vx, vz = features[..., 2].clone(), features[..., 3].clone()
    vx_0, vz_0 = vx[..., 0].clone(), vz[..., 0].clone()
    vx[..., 0], vz[..., 0] = 0, 0
    tx, tz = features[..., 4].clone(), features[..., 5].clone()
    tx_0, tz_0 = tx[..., 0].clone(), tz[..., 0].clone()
    features[..., 0] = sin * cos_0 - cos * sin_0
    features[..., 1] = cos * cos_0 + sin * sin_0
    features[..., 2] = vx * cos_0 - vz * sin_0
    features[..., 3] = vx * sin_0 + vz * cos_0
    features[..., 4] = tx * cos_0 - tz * sin_0
    features[..., 5] = tx * sin_0 + tz * cos_0
    return features, (sin_0, cos_0, vx_0, vz_0, tx_0, tz_0)


def recover_coordinate_features(features, transformation):
    sin_0, cos_0, vx_0, vz_0, tx_0, tz_0 = transformation
    if isinstance(features, np.ndarray):
        features = features.copy()
        sin, cos = features[..., 0].copy(), features[..., 1].copy()
        vx, vz = features[..., 2].copy(), features[..., 3].copy()
        tx, tz = features[..., 4].copy(), features[..., 5].copy()
    else:
        features = features.clone()
        sin, cos = features[..., 0].clone(), features[..., 1].clone()
        vx, vz = features[..., 2].clone(), features[..., 3].clone()
        tx, tz = features[..., 4].clone(), features[..., 5].clone()

    features[..., 0] = sin * cos_0 + cos * sin_0
    features[..., 1] = cos * cos_0 - sin * sin_0
    features[..., 2] = vx * cos_0 + vz * sin_0
    features[..., 3] = -vx * sin_0 + vz * cos_0
    features[..., 0, 2] = vx_0
    features[..., 0, 3] = vz_0
    features[..., 4] = tx * cos_0 + tz * sin_0
    features[..., 5] = -tx * sin_0 + tz * cos_0
    return features
