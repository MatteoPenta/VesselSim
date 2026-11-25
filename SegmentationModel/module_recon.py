import logging
import os
import torch
import lightning
import numpy as np
import torch.nn.functional as F

from monai.transforms import RandCoarseDropoutd
from monai.inferers.inferer import SlidingWindowInfererAdapt

logger = logging.getLogger(__name__)


class PLModuleRecon(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss,
        optimizer_factory,
        prediction_threshold: float,
        recon_lambda: float = 0.1,
        recon_loss: str = "l1",
        recon_only_holes: bool = True,
        dropout_prob: float = 0.8,
        holes: int = 4,
        spatial_size=(1, 2, 2, 2),
        max_spatial_size=(1,8,8,8),
        fill_value: float = -1.0,
        scheduler_configs=None,
        evaluator=None,
    ):
        super().__init__()
        self._viz_last_triplet = None
        self.model = model
        self.loss = loss
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs
        self.prediction_threshold = prediction_threshold
        self.rank = 0 if "LOCAL_RANK" not in os.environ else os.environ["LOCAL_RANK"]
        self.evaluator = evaluator

        self.recon_lambda = float(recon_lambda)
        self.recon_loss = str(recon_loss).lower()
        self.recon_only_holes = bool(recon_only_holes)

        self.fill_value = float(fill_value)
        self.coarse = RandCoarseDropoutd(
            keys=["image"],
            prob=float(dropout_prob),
            holes=int(holes),
            spatial_size=tuple(spatial_size),  # min size
            max_spatial_size=tuple(max_spatial_size),  # max size
            fill_value=self.fill_value,
        )
        logger.info(f"RandCoarseDropoutd spatial_size={spatial_size}, "
                    f"max_spatial_size={max_spatial_size}, holes={holes}")

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(params=self.parameters())
        if self.scheduler_configs is not None:
            schedulers = []
            for scheduler_name, scheduler_config in self.scheduler_configs.items():
                if scheduler_config is None:
                    continue
                scheduler_config["scheduler"] = scheduler_config["scheduler"](optimizer=optimizer)
                schedulers.append(dict(scheduler_config))
            return [optimizer], schedulers
        return optimizer

    def _recon_criterion(self, recon, target):
        if self.recon_loss == "mse":
            return F.mse_loss(recon, target)
        return F.l1_loss(recon, target)

    def _apply_corruption_with_mask(self, clean):
        corrupt = self.coarse({"image": clean})["image"]
        hole = (corrupt != clean)
        return corrupt, hole

    def _masked_recon_loss(self, recon, target, hole):
        hole = hole.to(recon.dtype)
        num = torch.sum(torch.abs(recon - target) * hole)
        den = torch.sum(hole) + 1e-6
        return num / den

    def training_step(self, batch, batch_idx):
        image, mask, gt_cache = batch
        if gt_cache is None:
            pass
        elif hasattr(gt_cache, "as_tensor"):
            gt_cache = gt_cache.as_tensor()
        elif torch.is_tensor(gt_cache):
            pass
        else:
            gt_cache = torch.as_tensor(gt_cache)

        clean = image
        corrupt, hole = self._apply_corruption_with_mask(clean)
        seg_logits, recon = self.model(corrupt, return_recon=True)

        if hasattr(self.loss, "loss_type") and self.loss.loss_type == "cbdice":
            seg_loss = self.loss(seg_logits, mask.float(), gt_cache=gt_cache)
        else:
            seg_loss = self.loss(seg_logits, mask.float())

        if self.recon_only_holes:
            rec_loss = self._masked_recon_loss(recon, clean, hole)
        else:
            rec_loss = self._recon_criterion(recon, clean)

        loss = seg_loss + self.recon_lambda * rec_loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_seg_loss", seg_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_rec_loss", rec_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self._viz_last_triplet = (clean, corrupt, recon)
        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            image, mask, gt_cache, name = batch
        else:
            image, mask, gt_cache = batch
            name = [f"drand"]

        if gt_cache is None:
            pass
        elif hasattr(gt_cache, "as_tensor"):
            gt_cache = gt_cache.as_tensor()
        elif torch.is_tensor(gt_cache):
            pass
        else:
            gt_cache = torch.as_tensor(gt_cache)

        seg_logits = self.model(image, return_recon=False)

        if hasattr(self.loss, "loss_type") and self.loss.loss_type == "cbdice":
            loss = self.loss(seg_logits, mask.float(), gt_cache=gt_cache)
        else:
            loss = self.loss(seg_logits, mask.float())

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        metrics = self.evaluator.estimate_metrics(
            seg_logits.sigmoid().squeeze(), mask.squeeze(),
            threshold=self.prediction_threshold, fast=True
        )
        for metric, value in metrics.items():
            value = value.item() if isinstance(value, (torch.Tensor, np.ndarray)) else value
            self.log(f"val_{name[0]}_{metric}", value, logger=(self.rank == 0))

class PLModuleReconFinetune(PLModuleRecon):
    def __init__(
        self,
        dataset_name: str = None,
        input_size: tuple = None,
        batch_size: int = None,
        num_shots: int = None,
        *args,
        **kwargs,
    ):
        self.dataset_name = dataset_name
        super().__init__(*args, **kwargs)

        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=input_size, sw_batch_size=batch_size, overlap=0.5
        )
        self.num_shots = num_shots

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            image, mask, gt_cache = batch
        elif len(batch) == 4:
            image, mask, gt_cache, _fourth = batch
        elif len(batch) == 5:
            image, mask, gt_cache, _fourth, _name = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        if gt_cache is None:
            pass
        elif hasattr(gt_cache, "as_tensor"):
            gt_cache = gt_cache.as_tensor()
        elif torch.is_tensor(gt_cache):
            pass
        else:
            gt_cache = torch.as_tensor(gt_cache)

        clean = image

        if float(self.recon_lambda) == 0.0:
            seg_logits = self.model(clean, return_recon=False)

            if hasattr(self.loss, "loss_type") and self.loss.loss_type == "cbdice":
                seg_loss = self.loss(seg_logits, mask.float(), gt_cache=gt_cache)
            else:
                seg_loss = self.loss(seg_logits, mask.float())

            loss = seg_loss
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train_seg_loss", seg_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
            # keep train_rec_loss at 0 for clean logs
            self.log(
                "train_rec_loss",
                torch.zeros((), device=seg_logits.device, dtype=seg_logits.dtype),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
            self._viz_last_triplet = None
            return loss

        # Otherwise: corruption + recon head + weighted loss
        clean = image
        corrupt, hole = self._apply_corruption_with_mask(clean)
        seg_logits, recon = self.model(corrupt, return_recon=True)

        if hasattr(self.loss, "loss_type") and self.loss.loss_type == "cbdice":
            seg_loss = self.loss(seg_logits, mask.float(), gt_cache=gt_cache)
        else:
            seg_loss = self.loss(seg_logits, mask.float())

        if self.recon_only_holes:
            rec_loss = self._masked_recon_loss(recon, clean, hole)
        else:
            rec_loss = self._recon_criterion(recon, clean)

        loss = seg_loss + float(self.recon_lambda) * rec_loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_seg_loss", seg_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_rec_loss", rec_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        self._viz_last_triplet = (clean, corrupt, recon)
        return loss

    def validation_step(self, batch, batch_idx):
        name = [self.dataset_name or "drand"]

        if len(batch) == 3:
            image, mask, gt_cache = batch
        elif len(batch) == 4:
            image, mask, gt_cache, fourth = batch
            if isinstance(fourth, (list, tuple)) and len(fourth) > 0 and isinstance(fourth[0], str):
                name = fourth
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")

        if gt_cache is None:
            pass
        elif hasattr(gt_cache, "as_tensor"):
            gt_cache = gt_cache.as_tensor()
        elif torch.is_tensor(gt_cache):
            pass
        else:
            gt_cache = torch.as_tensor(gt_cache)

        with torch.no_grad():
            seg_logits = self.sliding_window_inferer(
                image, lambda x: self.model(x, return_recon=False)
            )

            if hasattr(self.loss, "loss_type") and self.loss.loss_type == "cbdice":
                loss = self.loss(seg_logits, mask.float(), gt_cache=gt_cache)
            else:
                loss = self.loss(seg_logits, mask.float())

            self.log(f"{name[0]}_val_loss", loss.item())

            metrics = self.evaluator.estimate_metrics(
                seg_logits.sigmoid().squeeze(),
                mask.squeeze(),
                threshold=self.prediction_threshold,
                fast=True,
            )
            for m, v in metrics.items():
                v = v.item() if isinstance(v, (torch.Tensor, np.ndarray)) else v
                self.log(f"{name[0]}_val_{m}", v)
                if m == "dice":
                    self.log("val_DiceMetric", v)

        return loss
