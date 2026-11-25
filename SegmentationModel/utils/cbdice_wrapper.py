import torch
import torch.nn as nn
from .cbdice_loss import SoftcbDiceLoss


class SoftcbDiceLossWrapper(nn.Module):
    """
    Wrapper for binary UNet (1 output channel logits).
    Converts logits (B,1,...) -> 2-class logits (B,2,...) so SoftcbDiceLoss works.
    """
    def __init__(self, iter_=10, smooth=1.0, t_skeletonize_flage: bool = False):
        super().__init__()
        self.loss = SoftcbDiceLoss(iter_=iter_, smooth=smooth)
        self.t_skeletonize_flage = t_skeletonize_flage

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor, gt_cache=None) -> torch.Tensor:
        # y_true expected shape by SoftcbDiceLoss: (B,1,...) with {0,1}
        if y_true.ndim == logits.ndim - 1:
            y_true = y_true.unsqueeze(1)
        y_true = y_true.float()

        # If model already outputs multi-class logits, pass through
        if logits.shape[1] >= 2:
            return self.loss(logits, y_true, t_skeletonize_flage=self.t_skeletonize_flage, gt_cache=gt_cache)

        # Binary case: logits is (B,1,...) -> build 2-class logits
        l_fg = logits[:, 0:1, ...]
        l_bg = -l_fg
        logits_2ch = torch.cat([l_bg, l_fg], dim=1)  # (B,2,...)

        return self.loss(logits_2ch, y_true, t_skeletonize_flage=self.t_skeletonize_flage, gt_cache=gt_cache)
