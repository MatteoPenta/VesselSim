from pathlib import Path

import numpy as np
import torch
from matplotlib.mlab import detrend_none
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.measure import euler_number, label
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import SimpleITK as sitk
from torch.utils.data import Dataset


class Evaluator:
    def extract_labels(self, gt_array, pred_array):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L18.
        """
        labels_gt = np.unique(gt_array)
        labels_pred = np.unique(pred_array)
        labels = list(set().union(labels_gt, labels_pred))
        labels = [int(x) for x in labels]
        return labels


    def cl_dice(self, v_p, v_l):
        """
        Adapted from https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py.
        """

        def cl_score(v, s):
            denom = np.sum(s)
            if denom == 0:
                return 0.0
            return np.sum(v * s) / denom

        if np.sum(v_p) == 0 and np.sum(v_l) == 0:
            return np.nan

        if len(v_p.shape) == 2:
            tprec = cl_score(v_p, skeletonize(v_l))
            tsens = cl_score(v_l, skeletonize(v_p))
        elif len(v_p.shape) == 3:
            tprec = cl_score(v_p, skeletonize_3d(v_l))
            tsens = cl_score(v_l, skeletonize_3d(v_p))
        else:
            raise ValueError(f"Invalid shape for cl_dice: {v_p.shape}")
        return 2 * tprec * tsens / (tprec + tsens + np.finfo(float).eps)

    def estimate_metrics(self, pred_seg, gt_seg, threshold=0.5, fast=False):
        metrics = {}

        pred = (pred_seg >= threshold)
        gt = (gt_seg > 0.5)
        tp = (pred & gt).sum().item()
        fp = (pred & ~gt).sum().item()
        fn = (~pred & gt).sum().item()

        if fast:
            den_dice = (2 * tp + fp + fn)
            metrics["dice"] = 1.0 if den_dice == 0 else (2 * tp) / den_dice
            return metrics

        cldice = self.cl_dice(
            pred.squeeze().detach().to(torch.uint8).cpu().numpy(),
            gt.squeeze().detach().to(torch.uint8).cpu().numpy()
        )

        denom_dice = (2 * tp + fp + fn)
        if denom_dice == 0:
            metrics["dice"] = 1
        else:
            metrics["dice"] = (2 * tp) / denom_dice
        metrics["cldice"] = cldice

        return metrics


def calculate_mean_metrics(results, round_to=2):
    mean = {}
    for k in results[0].keys():
        numbers = [r[k] for r in results]
        numbers = [n for n in numbers if np.isnan(n) == False]
        mean[k] = np.mean(numbers)

        if "dice" in k:
            mean[k] = mean[k] * 100
        mean[k] = np.round(mean[k], round_to)
    return mean
