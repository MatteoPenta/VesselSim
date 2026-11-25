import logging

from monai import transforms
from monai.transforms import Compose

logger = logging.getLogger(__name__)
from typing import Optional

import importlib
from monai.transforms import Compose
import monai.transforms as monai_t

import importlib
from collections.abc import Mapping

import monai.transforms as monai_t
from monai.transforms import Compose
from omegaconf import OmegaConf


def _resolve_transform_ctor(name: str):
    if "." not in name:
        if not hasattr(monai_t, name):
            raise AttributeError(f"monai.transforms has no attribute '{name}'")
        return getattr(monai_t, name)

    module_path, attr = name.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    if not hasattr(mod, attr):
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr}'")
    return getattr(mod, attr)


def generate_transforms(transform_list):
    tfs = []

    for item in transform_list:
        # Accept dict or OmegaConf DictConfig
        if not isinstance(item, Mapping):
            raise ValueError(f"Each transform must be a mapping (dict/DictConfig), got: {type(item)} | {item}")

        # Convert DictConfig -> plain dict (safe)
        item = OmegaConf.to_container(item, resolve=True)

        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Each transform must be a single-key dict, got: {item}")

        transform_name, kwargs = next(iter(item.items()))
        kwargs = kwargs or {}

        ctor = _resolve_transform_ctor(transform_name)
        tfs.append(ctor(**kwargs))

    return Compose(tfs)