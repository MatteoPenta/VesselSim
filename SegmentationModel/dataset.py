import logging
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.io import determine_reader_writer
from utils.data import generate_transforms

logger = logging.getLogger(__name__)


class UnionDataset(Dataset):
    """
    Dataset that accumulates all given datasets.
    """

    def __init__(self, dataset_configs, mode, finetune=False): # mode is train, val, or finetune
        super().__init__()
        # init datasets
        self.finetune = finetune
        self.datasets, probs = [], []
        self.len = 0
        self.mode = mode
        for name, dataset_config in dataset_configs.items():
            data_dir = Path(dataset_config.path) / mode if finetune else Path(dataset_config.path)
            num_samples = 0
            while (data_dir / str(num_samples)).exists():
                num_samples += 1
            paths = [data_dir / str(i) for i in range(num_samples)]

            #### optional cbdice skeletonized cache directory
            cbdice_cache_dir = None
            if hasattr(dataset_config, "cbdice_cache_dir") and dataset_config.cbdice_cache_dir is not None:
                cbdice_cache_dir = Path(dataset_config.cbdice_cache_dir) / mode

            self.len += len(paths)
            self.datasets.append(
                {
                    "name": name,
                    "paths": paths,
                    "reader": determine_reader_writer(dataset_config.file_format)(),
                    "transforms": generate_transforms(dataset_config.transforms[mode]),
                    "sample_prop": dataset_config.sample_prop,
                    "filter_dataset_IDs": dataset_config.filter_dataset_IDs,
                    "cbdice_cache_dir": cbdice_cache_dir
                }
            )
            probs.append(dataset_config.sample_prop)

        # ensure that probs sum up to 1
        probs = torch.tensor(probs)
        self.probs = probs / probs.sum()

    def __len__(self):
        return self.len

    def _load_cache_if_configured(self, dataset: Dict[str, Any], sample_folder: Path) -> Optional[Dict[str, torch.Tensor]]:
        cache_root: Optional[Path] = dataset.get("cbdice_cache_dir", None)
        if cache_root is None:
            return None

        sample_name = sample_folder.name
        cache_path = cache_root / f"{sample_name}.pt"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"cbDice cache not found for sample '{sample_name}' (mode={self.mode}). "
                f"Expected: {cache_path}"
            )
        gt_cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(gt_cache, dict):
            raise TypeError(f"Expected gt_cache to be a dict, got {type(gt_cache)} from {cache_path}")

        return gt_cache

    def __getitem__(self, idx: int):
        # sample dataset
        dataset_id = torch.multinomial(self.probs, 1).item()
        dataset = self.datasets[dataset_id]

        # sample data sample
        while True:
            data_idx = idx if self.finetune else torch.randint(0, len(dataset["paths"]), (1,)).item()
            sample_id = dataset["paths"][data_idx]

            img_path = sample_id / "img.npy"
            mask_path = sample_id / "mask.npy"

            if dataset['filter_dataset_IDs'] is not None:
                if int(img_path.stem.split("_")[-1]) in dataset['filter_dataset_IDs']:
                    continue

            img = dataset['reader'].read_images(str(img_path))[0].astype(np.float32)
            mask = dataset['reader'].read_images(str(mask_path))[0].astype(bool)
            transformed = dataset['transforms']({'Image': img, 'Mask': mask})
            img_t = transformed['Image']
            mask_t = transformed['Mask'] > 0

            gt_cache = self._load_cache_if_configured(dataset, sample_id)
            if gt_cache is not None:
                q_vl, q_slvl, q_sl = gt_cache["q_vl"], gt_cache["q_slvl"], gt_cache["q_sl"]
                if q_vl.ndim == mask_t.ndim and q_vl.shape[0] == 1:
                    q_vl, q_slvl, q_sl = q_vl[0], q_slvl[0], q_sl[0]
                gt_cache_t = torch.stack([q_vl.float(), q_slvl.float(), q_sl.float()], dim=0)
            else:
                spatial = mask_t.shape[1:] if mask_t.ndim >= 4 else mask_t.shape[-2:]
                gt_cache_t = torch.zeros((3, *spatial), dtype=torch.float32)
                gt_cache_t.reshape(-1)[0] = -1.0

            return img_t, mask_t, gt_cache_t
