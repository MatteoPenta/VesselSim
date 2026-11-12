import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from .cbdice_wrapper import SoftcbDiceLossWrapper


class DiceCeCbdice(nn.Module):
    loss_type = "cbdice"
    def __init__(
        self,
        beta: float = 1.0,
        dicece_kwargs=None,
        cbdice_kwargs=None,
    ):
        super().__init__()
        dicece_kwargs = dicece_kwargs or {}
        cbdice_kwargs = cbdice_kwargs or {}

        self.base = DiceCELoss(**dicece_kwargs)
        self.cbdice = SoftcbDiceLossWrapper(**cbdice_kwargs)
        self.beta = beta

    def forward(self, logits: torch.Tensor, target: torch.Tensor, gt_cache=None) -> torch.Tensor:
        base_loss = self.base(logits, target)
        cbdice_loss = self.cbdice(logits, target, gt_cache=gt_cache)
        return base_loss + self.beta * cbdice_loss
