import logging
import warnings
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import hydra
import torch
import torch.utils
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger  # Removed WandbLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
import torch.multiprocessing as mp

from dataset import UnionDataset
from utils.evaluation import Evaluator

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class LossPlotterCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # 1) Find CSVLogger
        csv_logger = None
        if trainer.loggers:
            for lg in trainer.loggers:
                if isinstance(lg, CSVLogger):
                    csv_logger = lg
                    break
        if csv_logger is None:
            return

        metrics_path = os.path.join(csv_logger.log_dir, "metrics.csv")
        if not os.path.exists(metrics_path):
            return

        # 2) Read CSV
        try:
            metrics = pd.read_csv(metrics_path)
        except Exception:
            return

        if "epoch" not in metrics.columns:
            return

        # 3) Aggregate per epoch
        agg_dict = {}
        for col in ["train_loss_epoch", "val_loss", "val_drand_dice", "train_seg_loss_epoch", "train_rec_loss_epoch"]:
            if col in metrics.columns:
                agg_dict[col] = "mean"
        if not agg_dict:
            return

        try:
            epoch_metrics = metrics.groupby("epoch").agg(agg_dict).reset_index()
        except Exception:
            return

        # Loss curve (train_loss vs val_loss) ----
        plt.figure(figsize=(10, 6))
        if "train_loss_epoch" in epoch_metrics.columns:
            td = epoch_metrics.dropna(subset=["train_loss_epoch"])
            if not td.empty:
                plt.plot(td["epoch"], td["train_loss_epoch"], label="Training Loss (Avg)", marker="o")
        if "val_loss" in epoch_metrics.columns:
            vd = epoch_metrics.dropna(subset=["val_loss"])
            if not vd.empty:
                plt.plot(vd["epoch"], vd["val_loss"], label="Validation Loss", linewidth=2, marker="o")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Progress (Epoch {trainer.current_epoch})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(csv_logger.log_dir, "loss_curve.png"))
        plt.close()

        # Recon losses (train_seg_loss vs train_rec_loss) ----
        if ("train_seg_loss_epoch" in epoch_metrics.columns) or ("train_rec_loss_epoch" in epoch_metrics.columns):
            plt.figure(figsize=(10, 6))
            if "train_seg_loss_epoch" in epoch_metrics.columns:
                sd = epoch_metrics.dropna(subset=["train_seg_loss_epoch"])
                if not sd.empty:
                    plt.plot(sd["epoch"], sd["train_seg_loss_epoch"], label="Train Seg Loss (Avg)", marker="o")
            if "train_rec_loss_epoch" in epoch_metrics.columns:
                rd = epoch_metrics.dropna(subset=["train_rec_loss_epoch"])
                if not rd.empty:
                    plt.plot(rd["epoch"], rd["train_rec_loss_epoch"], label="Train Recon Loss (Avg)", marker="o")

            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"Seg vs Recon Loss (Epoch {trainer.current_epoch})")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig(os.path.join(csv_logger.log_dir, "recon_losses.png"))
            plt.close()

        # C) Dice curve ----
        if "val_drand_dice" in epoch_metrics.columns:
            dice_data = epoch_metrics.dropna(subset=["val_drand_dice"])
            if not dice_data.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(dice_data["epoch"], dice_data["val_drand_dice"], label="Validation Dice", marker="o")
                plt.xlabel("Epochs")
                plt.ylabel("Dice Score")
                plt.title(f"Validation Dice (Epoch {trainer.current_epoch})")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.savefig(os.path.join(csv_logger.log_dir, "dice_curve.png"))
                plt.close()

class ReconVizCallback(Callback):
    def __init__(self, num_examples: int = 3, fill_value: float = -1.0):
        super().__init__()
        self.num_examples = int(num_examples)
        self.fill_value = float(fill_value)
        self._cache = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._cache = []

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank != 0:
            return

        if not hasattr(pl_module, "_viz_last_triplet"):
            return

        clean, corrupt, recon = pl_module._viz_last_triplet  # tensors on GPU/CPU
        if clean is None:
            return
        if clean.ndim != 5:
            return

        hole = (corrupt == self.fill_value)

        B = clean.shape[0]
        for b in range(B):
            if len(self._cache) >= self.num_examples:
                return
            if hole[b].any():
                # store only 1st channel for viz (assumes C=1)
                self._cache.append((
                    clean[b:b+1].detach().cpu(),
                    corrupt[b:b+1].detach().cpu(),
                    recon[b:b+1].detach().cpu(),
                ))

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return
        if len(self._cache) == 0:
            return

        # Find CSVLogger folder
        csv_logger = None
        if trainer.loggers:
            for lg in trainer.loggers:
                if isinstance(lg, CSVLogger):
                    csv_logger = lg
                    break
        if csv_logger is None:
            return

        epoch = trainer.current_epoch
        save_path = os.path.join(csv_logger.log_dir, f"reconstruction_epoch_{epoch:03d}.png")

        n = min(len(self._cache), self.num_examples)

        fig, axes = plt.subplots(nrows=n, ncols=3, figsize=(12, 4 * n))
        if n == 1:
            axes = np.expand_dims(axes, axis=0)  # make [1,3]

        for i in range(n):
            clean, corrupt, recon = self._cache[i]
            clean = clean[0, 0]
            corrupt = corrupt[0, 0]
            recon = recon[0, 0]
            hole = (corrupt == self.fill_value)
            z_any = hole.any(dim=0).any(dim=0)  # [D]
            z_idxs = torch.where(z_any)[0].numpy() if z_any.any() else None
            if z_idxs is not None and len(z_idxs) > 0:
                z = int(z_idxs[len(z_idxs) // 2])
            else:
                z = int(clean.shape[-1] // 2)

            clean2d = clean[:, :, z].numpy()
            corrupt2d = corrupt[:, :, z].numpy()
            recon2d = recon[:, :, z].numpy()

            axes[i, 0].imshow(clean2d, cmap="gray")
            axes[i, 0].set_title(f"Clean (ex {i}) z={z}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(corrupt2d, cmap="gray")
            axes[i, 1].set_title("Corrupted (coarse dropout)")
            axes[i, 1].axis("off")

            vmin, vmax = clean2d.min(), clean2d.max()
            axes[i, 2].imshow(recon2d, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, 2].set_title("Reconstruction")
            axes[i, 2].axis("off")
            print("recon min/max:", recon.min().item(), recon.max().item())

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)


class KeepOnlyNFiles(Callback):
    def __init__(self, dirpath: str, pattern: str, keep: int = 3):
        self.dirpath = Path(dirpath)
        self.pattern = pattern
        self.keep = keep

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0:
            return

        if not self.dirpath.exists():
            return

        files = sorted(
            self.dirpath.glob(self.pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old in files[self.keep:]:
            old.unlink(missing_ok=True)

@hydra.main(config_path="configs", config_name="train_recon", version_base="1.3.2")
def main(cfg):
    seed_everything(cfg.seed, True)
    torch.set_float32_matmul_precision("medium")
    run_name = 'pretrain_' + cfg.run_name

    # 1. Init CSV Logger
    csv_logger = CSVLogger(
        save_dir=cfg.chkpt_folder,
        name="csv_logs",
        version=run_name
    )

    # 2. Callbacks
    lr_monitor = LearningRateMonitor()
    monitor_metric = "val_loss"
    plotter_callback = LossPlotterCallback()
    recon_viz_callback = ReconVizCallback(num_examples=3, fill_value=0.0)

    checkpoint_dir = os.path.join(cfg.chkpt_folder, "checkpoints", cfg.run_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor=monitor_metric,
        save_top_k=3,
        mode="min",
        filename=(f"{run_name}_" + "{step}_{" + monitor_metric + ":.2f}"),
        auto_insert_metric_name=True,
        save_last=True
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "="
    checkpoint_callback.CHECKPOINT_NAME_LAST = run_name + "_last"
    progress_bar = TQDMProgressBar(refresh_rate=100)

    # 3. Init Trainer
    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": [csv_logger],
        "callbacks": [lr_monitor, checkpoint_callback, plotter_callback, recon_viz_callback ,progress_bar],
        "devices": cfg.devices,
        "enable_progress_bar": True
    }
    trainer = trainer(**trainer_additional_kwargs)

    # 4. Init DataLoader
    train_dataset = UnionDataset(cfg.data, "train")
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset)
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")

    val_transforms_resolved = OmegaConf.to_container(
        cfg.data.MyDataset.transforms.val,
        resolve=True
    )
    if hasattr(cfg.data.MyDataset, "cbdice_cache_dir"):
        cbdice_dir = cfg.data.MyDataset.cbdice_cache_dir
    else:
        cbdice_dir = None
    val_data_config = OmegaConf.create({
        "MyValDataset": {
            "path": cfg.eval_path,
            "sample_prop": 1.0,
            "filter_dataset_IDs": None,
            "file_format": "npy",
            "cbdice_cache_dir": cbdice_dir,
            "transforms": {
                "val": val_transforms_resolved
            }
        }
    })
    val_dataset = UnionDataset(val_data_config, "val", finetune=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
        num_workers=0
    )
    logger.info(f"Val dataset size mapped to {len(val_dataset)} samples")

    # 5. Init Model
    model = hydra.utils.instantiate(cfg.model)
    if cfg.path_to_chkpt is not None:
        chkpt = torch.load(cfg.path_to_chkpt, map_location=f'cuda:{cfg.devices[0]}')
        model_chkpt = {k.replace("model.", ""): e for k, e in chkpt["state_dict"].items() if "model" in k}
        model.load_state_dict(model_chkpt)


    # 6. Init Lightning Module
    evaluator = Evaluator()
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, evaluator=evaluator
    )

    # 7. Train Loop
    logger.info("Starting training")
    trainer.fit(lightning_module, train_loader, val_loader)
    logger.info("Finished training")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()