import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist
import SimpleITK as sitk

from monai.transforms import RandCoarseDropoutd

from utils.data import generate_transforms
from utils.io import determine_reader_writer

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# -------------------------
# Helpers
# -------------------------
def pad_to_min_patch_size(vol_b1xyz: torch.Tensor, patch_size, mode="reflect"):
    """
    vol_b1xyz: [1,1,D,H,W]
    patch_size: (pd,ph,pw)

    Pads volume if any dimension is smaller than patch_size.
    Returns padded volume (same device).
    """
    assert vol_b1xyz.ndim == 5
    _, _, D, H, W = vol_b1xyz.shape
    pd, ph, pw = patch_size

    pad_d = max(pd - D, 0)
    pad_h = max(ph - H, 0)
    pad_w = max(pw - W, 0)

    if pad_d == 0 and pad_h == 0 and pad_w == 0:
        return vol_b1xyz

    pad = [
        pad_w // 2, pad_w - pad_w // 2,  # W
        pad_h // 2, pad_h - pad_h // 2,  # H
        pad_d // 2, pad_d - pad_d // 2,  # D
    ]
    return F.pad(vol_b1xyz, pad, mode=mode)


def load_model(cfg, device: torch.device):
    local_path = Path(cfg.ckpt_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.ckpt_path}")

    logger.info(f"Loading model from {cfg.ckpt_path}.")
    ckpt = torch.load(str(local_path), map_location=device, weights_only=False)
    if "state_dict" in ckpt:
        logger.info("Detected PyTorch Lightning checkpoint. Extracting state_dict...")
        ckpt = ckpt["state_dict"]

    # Strip Lightning prefix model.
    stripped = {}
    for k, v in ckpt.items():
        stripped[k[len("model.") :]] = v if k.startswith("model.") else v

    model = hydra.utils.instantiate(cfg.model)

    mapped = {}
    for k, v in stripped.items():
        if k.startswith(("seg.", "recon.")):
            mapped[k] = v
        else:
            mapped["seg." + k] = v

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        logger.info(f"Missing keys (expected for recon head): {missing[:10]} ...")
    if unexpected:
        logger.info(f"Unexpected keys: {unexpected[:10]} ...")

    return model


def get_paths(cfg):
    image_paths = sorted([p for p in Path(cfg.image_path).iterdir() if p.is_file()])
    if cfg.mask_path:
        mask_paths = [Path(cfg.mask_path) / p.name for p in image_paths]
        assert all(mp.exists() for mp in mask_paths), "All mask paths must exist and match image names."
    else:
        mask_paths = None
    return image_paths, mask_paths


def resample(image, factor=None, target_shape=None):
    if factor == 1 or factor is None:
        return image
    if target_shape is not None:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


def masked_l1_on_holes(recon, target, corrupt):
    hole = (corrupt != target).to(recon.dtype)
    den = hole.sum()
    if den < 1.0:
        return F.l1_loss(recon, target)
    return (torch.abs(recon - target) * hole).sum() / (den + 1e-6)


def sample_random_patches_3d(vol_b1dhw: torch.Tensor, patch_size, n_patches: int, rng: np.random.Generator):
    assert vol_b1dhw.ndim == 5 and vol_b1dhw.shape[0] == 1 and vol_b1dhw.shape[1] == 1
    D, H, W = vol_b1dhw.shape[2], vol_b1dhw.shape[3], vol_b1dhw.shape[4]
    pd, ph, pw = patch_size

    if D < pd or H < ph or W < pw:
        raise ValueError(f"Patch {patch_size} bigger than volume {(D, H, W)}")

    patches = []
    for _ in range(n_patches):
        d0 = int(rng.integers(0, D - pd + 1))
        h0 = int(rng.integers(0, H - ph + 1))
        w0 = int(rng.integers(0, W - pw + 1))
        p = vol_b1dhw[:, :, d0 : d0 + pd, h0 : h0 + ph, w0 : w0 + pw]  # [1,1,pd,ph,pw]
        patches.append(p.squeeze(0))  # -> [1,pd,ph,pw]
    return torch.stack(patches, dim=0)  # [n,1,pd,ph,pw]


def set_requires_grad_for_tta(model: torch.nn.Module, tta_on: bool = True):
    """Adapt the shared DynUNet encoder; keep both decoders frozen."""
    for p in model.parameters():
        p.requires_grad = False

    if not tta_on:
        return

    ### ALL ENCODER PARAMS
    encoder_prefixes = (
        "seg.input_block",
        "seg.downsamples",
        "seg.bottleneck",
    )
    for name, p in model.named_parameters():
        if name.startswith(encoder_prefixes):
            p.requires_grad = True


def restore_from_base_state_(model: torch.nn.Module, base_state_cpu: dict):
    with torch.no_grad():
        sd = model.state_dict()
        for k, dst in sd.items():
            src_cpu = base_state_cpu[k]
            src = src_cpu.to(device=dst.device, dtype=dst.dtype, non_blocking=True)
            dst.copy_(src)


@hydra.main(config_path="configs", config_name="inference_recon", version_base="1.3.2")
def main(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}.")

    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    transforms = generate_transforms(cfg.transforms_config)

    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    image_paths, mask_paths = get_paths(cfg)
    logger.info(f"Found {len(image_paths)} images in {cfg.image_path}.")

    file_ending = (cfg.image_file_ending if cfg.image_file_ending else image_paths[0].suffix)
    image_reader_writer = determine_reader_writer(file_ending)()

    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap,
        mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
    )

    # --- TTA Recon config ---
    tta_rec = cfg.tta_recon
    fill_value = float(tta_rec.fill_value)

    coarse = RandCoarseDropoutd(
        keys=["image"],
        prob=float(tta_rec.dropout_prob),
        holes=int(tta_rec.holes),
        spatial_size=tuple(tta_rec.spatial_size),
        max_spatial_size=tuple(tta_rec.max_spatial_size),
        fill_value=fill_value,
    )

    patch_batch_size = int(getattr(tta_rec, "patch_batch_size", 4))
    use_amp = bool(getattr(tta_rec, "use_amp", True))

    for idx, image_path in tqdm(list(enumerate(image_paths)), total=len(image_paths), desc="Processing images"):
        img_np = image_reader_writer.read_images(image_path)[0].astype(np.float32)
        image = transforms(img_np)[None].to(device)

        if image.ndim == 4:
            image = image.unsqueeze(1)

        if mask_paths:
            mask_np = image_reader_writer.read_images(mask_paths[idx])[0]
            mask = torch.as_tensor(mask_np, device=device).bool()
            if mask.ndim == 3:
                mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            mask = None

        # ---------- optional TTA pre-ops ----------
        if cfg.tta.invert:
            image = 1 - image if image.mean() > cfg.tta.invert_mean_thresh else image

        if cfg.tta.equalize_hist:
            image_np2 = image.detach().cpu().squeeze().numpy()
            image_equal_hist_np = equalize_hist(image_np2, nbins=cfg.tta.hist_bins)
            image = torch.from_numpy(image_equal_hist_np).to(device)[None][None]  # [1,1,D,H,W]

        # ---------- per-volume reconstruction TTA ----------
        if bool(tta_rec.apply):
            restore_from_base_state_(model, base_state)

            set_requires_grad_for_tta(model, tta_on=True)
            model.train()

            optim = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=float(tta_rec.lr)
            )

            scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
            rng = np.random.default_rng(int(tta_rec.seed) + idx)

            steps = int(tta_rec.steps)
            total_patches = int(tta_rec.num_patches)
            tta_patch_size = tuple(tta_rec.patch_size)

            for _step in range(steps):
                remaining = total_patches
                while remaining > 0:
                    n = min(patch_batch_size, remaining)
                    remaining -= n

                    image_padded = pad_to_min_patch_size(image, tta_patch_size, mode="reflect")
                    patches = sample_random_patches_3d(
                        image_padded,
                        patch_size=tta_patch_size,
                        n_patches=n,
                        rng=rng
                    )

                    corrupt = coarse({"image": patches})["image"]
                    optim.zero_grad(set_to_none=True)

                    if scaler.is_enabled():
                        with torch.cuda.amp.autocast():
                            _seg, recon = model(corrupt, return_recon=True)
                            rec_loss = masked_l1_on_holes(recon, patches, corrupt)
                        scaler.scale(rec_loss).backward()
                        scaler.step(optim)
                        scaler.update()
                    else:
                        _seg, recon = model(corrupt, return_recon=True)
                        rec_loss = masked_l1_on_holes(recon, patches, corrupt)
                        rec_loss.backward()
                        optim.step()

                    del patches, corrupt, recon, rec_loss

            model.eval()
            set_requires_grad_for_tta(model, tta_on=False)

        # ---------- segmentation inference ----------
        with torch.no_grad():
            acc = None
            for scale in cfg.tta.scales:
                original_shape = image.shape
                img_s = resample(image, factor=scale)

                logits = inferer(img_s, lambda x: model(x, return_recon=False))
                logits = resample(logits, target_shape=original_shape)

                if acc is None:
                    acc = logits
                else:
                    if cfg.merging.max:
                        acc = torch.maximum(acc, logits)
                    else:
                        acc = acc + logits

            if not cfg.merging.max:
                acc = acc / float(len(cfg.tta.scales))

            pred = acc.sigmoid()

        pred_bin = (pred > cfg.merging.threshold)
        pred_bin_np = pred_bin.detach().to(torch.uint8).cpu().squeeze().numpy()

        if pred_bin_np.ndim != 3:
            raise RuntimeError(f"Expected 3D pred_bin_np after squeeze, got shape {pred_bin_np.shape}")

        if cfg.post.apply:
            pred_bin_np = remove_small_objects(
                pred_bin_np.astype(bool),
                min_size=cfg.post.small_objects_min_size,
                connectivity=cfg.post.small_objects_connectivity
            ).astype(np.uint8)

        output_filename = output_folder / f"{image_path.name.split('.')[0]}_{cfg.file_app}pred.{file_ending}"
        reference_img = sitk.ReadImage(str(image_path))
        output_sitk = sitk.GetImageFromArray(pred_bin_np.astype(np.uint8))
        output_sitk.CopyInformation(reference_img)
        sitk.WriteImage(output_sitk, str(output_filename))

        # ---------- restore base weights for next case ----------
        if bool(tta_rec.apply):
            restore_from_base_state_(model, base_state)
            model.eval()

        if device.type == "cuda":
            torch.cuda.empty_cache()
    logger.info("Done.")


if __name__ == "__main__":
    main()