import torch as th
from dataset.humanml3d import HumanML3DT2M, HumanML3DM2M
from dataset.interhuman import InterHumanM2M, InterHumanT2M
from loguru import logger
from typing import Literal
import numpy as np


class M2MDataset(th.utils.data.Dataset):
    def __init__(
        self,
        window_size: int = 64,
        window_stride: int = 10,
        mode: str = "train",
        feature_type: str = "264",
        split_features: bool = False,
        random_rotate: bool = False,
    ):
        kwargs = {
            "mode": mode,
            "window_size": window_size,
            "window_stride": window_stride,
            "feature_type": feature_type,
            "split_features": split_features,
        }
        self.interhuman = InterHumanM2M(**kwargs)
        self.humanml3d = HumanML3DM2M(**kwargs)
        self.n_samples = len(self.interhuman) + len(self.humanml3d)
        logger.info(f"M2MDataset: total number of samples {self.n_samples}")

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[tuple[str, str], int, np.ndarray]:
        if idx < len(self.interhuman):
            return self.interhuman[idx]
        else:
            return self.humanml3d[idx - len(self.interhuman)]


class T2MDataset(th.utils.data.Dataset):
    def __init__(
        self,
        mode: str = "train",
        feature_type: Literal[262, 264, 266] = 264,
        max_length: int = 300,
        random_rotate: bool = False,
    ):
        self.max_length = max_length
        self.humanml3d = HumanML3DT2M(
            mode=mode, max_length=max_length, feature_type=feature_type, random_rotate=random_rotate
        )
        self.interhuman = InterHumanT2M(
            mode=mode, max_length=max_length, feature_type=feature_type, random_rotate=random_rotate
        )
        self.feature_type = feature_type

        self.single_length = len(self.humanml3d)
        self.double_length = len(self.interhuman)

        logger.info(f"Total number of motions {self.single_length + self.double_length}")

    def __len__(self) -> int:
        return self.single_length + self.double_length

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, np.ndarray, int, int]:
        if idx < self.single_length:
            return self.humanml3d[idx]
            # text, motion, length = self.humanml3d[idx]
            # return text, motion, np.zeros_like(motion), length, 0
        else:
            text, motion1, motion2, length1, length2 = self.interhuman[idx - self.single_length]
            return text, motion1, motion2, length1, length2
