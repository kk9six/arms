import numpy as np
import torch


class OpsMeta(type):
    def __getattr__(cls, name):
        if name.startswith("_"):
            raise AttributeError(f"type object '{cls.__name__}' has no attribute '{name}'")

        def universal_op(*args, **kwargs):
            backend = cls.get_backend()
            if backend == np and "dim" in kwargs:
                kwargs["axis"] = kwargs.pop("dim")
            op = getattr(backend, name)
            return op(*args, **kwargs)

        setattr(cls, name, staticmethod(universal_op))
        return getattr(cls, name)


class _ops_numpy(metaclass=OpsMeta):
    @classmethod
    def get_backend(cls):
        return np


class _ops_torch(metaclass=OpsMeta):
    @classmethod
    def get_backend(cls):
        return torch


def get_ops(data: np.ndarray | torch.Tensor):
    """Get ops based on data type."""
    if isinstance(data, torch.Tensor):
        return _ops_torch
    elif isinstance(data, np.ndarray):
        return _ops_numpy
    else:
        raise TypeError(f"Unsupported type: {type(data)}")
