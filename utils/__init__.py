import numpy as np
import sys
import datetime
import random
import torch
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from einops import einsum, parse_shape, repeat


def matvec(matrix, vector, matrix_shape="", vector_shape=""):
    """
    Flexible matrix-vector multiplication with automatic broadcasting.
    Supports both NumPy and PyTorch tensors.

    Parameters
    ----------
    :param matrix: array_like
    :param vector: array_like
    :param matrix_shape: str, e.g. "" "t" "b t"
    :param vector_shape: str, e.g. "" "t" "b t"

    Returns
    -------
    :return: array_like
    """
    mat_shape = matrix.shape
    vec_shape = vector.shape
    assert len(mat_shape) >= 2, f"Matrix must be at least 2D, got shape {mat_shape}"
    assert len(vec_shape) >= 1, f"Vector must be at least 1D, got shape {vec_shape}"
    assert vec_shape[-1] == mat_shape[-1], (
        f"Vector last dim {vec_shape[-1]} must match matrix last dim {mat_shape[-1]}"
    )
    result = einsum(matrix, vector, f"{matrix_shape} m n, {vector_shape} n -> {vector_shape} m")
    return result


def inv_transform(R, t, R_shape="", t_shape=""):
    if isinstance(R, np.ndarray):
        shape = tuple(list(range(len(R.shape) - 2))) + (-1, -2)
        R_inv = np.transpose(R, shape)
        t_inv = matvec(R_inv, -t, R_shape, t_shape)
    elif isinstance(R, torch.Tensor):
        R_inv = R.transpose(-1, -2)
        t_inv = matvec(R_inv, -t, R_shape, t_shape)
    else:
        raise ValueError(f"Unsupported type: {type(R)}")
    return R_inv, t_inv


def rigid_transform(
    motion: np.ndarray | torch.Tensor,
    R: np.ndarray | torch.Tensor,
    t: np.ndarray | torch.Tensor,
    motion_shape: str,  # "" "t" "b t"
    R_shape: str,
    t_shape: str,
) -> np.ndarray | torch.Tensor:
    if isinstance(motion, np.ndarray):
        def concat_fn(x, dim=-1):
            return np.concatenate(x, axis=dim)
    else:
        def concat_fn(x, dim=-1):
            return torch.cat(x, dim=dim)

    motion_shape = f"{motion_shape} j"

    shape = motion.shape[:-1]
    joints = motion[..., : 22 * 3].reshape(shape + (22, 3))
    joints_transformed = matvec(R, joints, R_shape, motion_shape) + repeat(
        t, f"{t_shape} n -> {motion_shape} n", **(parse_shape(joints, f"{motion_shape} n"))
    )
    joints_transformed = joints_transformed.reshape(shape + (22 * 3,))
    if motion.shape[-1] == 66:
        return joints_transformed

    else:
        joint_vels = motion[..., 22 * 3 : 22 * 6].reshape(shape + (22, 3))
        rest = motion[..., 22 * 6 :]
        joint_vels_transformed = matvec(R, joint_vels, R_shape, motion_shape)
        joint_vels_transformed = joint_vels_transformed.reshape(shape + (22 * 3,))
        result = concat_fn([joints_transformed, joint_vels_transformed, rest], dim=-1)
        return result


def seed_everything(seed: int = 42):
    assert isinstance(seed, int), "seed must be an integer"
    logger.info(f"Global seed set to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(out_dir=None, log_file="run.log", level="INFO", mode="w"):
    logger.remove()
    if out_dir is not None:
        logger.add(Path(out_dir) / log_file, enqueue=True, mode=mode, level=level)
    # logger.add(sys.stdout, colorize=True, enqueue=True, level=level)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, enqueue=True, level=level)
    return logger.opt(colors=True)


def set_logger(out_dir=None, log_file="run.log", level="INFO", mode="w"):
    logger.remove()
    if out_dir is not None:
        logger.add(Path(out_dir) / log_file, enqueue=True, mode=mode, level=level)
    # logger.add(sys.stdout, colorize=True, enqueue=True)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, enqueue=True, level=level)


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
