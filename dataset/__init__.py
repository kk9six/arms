from .interhuman import (
    InterHumanT2M,
    InterHumanM2M,
)
from .humanml3d import HumanML3DM2M, HumanML3DT2M
from .dataset import M2MDataset, T2MDataset
from .interx import InterXM2M, InterXT2M, InterXT2MNative

build_dataset = {
    "InterHumanM2M": InterHumanM2M,
    "HumanML3DM2M": HumanML3DM2M,
    "M2MDataset": M2MDataset,
    "T2MDataset": T2MDataset,
    "InterHumanT2M": InterHumanT2M,
    "InterXM2M": InterXM2M,
    "InterXT2M": InterXT2M,
    "InterXT2MNative": InterXT2MNative,
    "HumanML3DT2M": HumanML3DT2M,
}
