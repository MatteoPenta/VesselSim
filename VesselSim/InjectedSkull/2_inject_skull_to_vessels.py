import numpy as np
import os
import random
import glob
import argparse
from pathlib import Path
import scipy.ndimage as ndimage
import yaml


def get_random_skull(skull_root_dir, skull_ids):
    # 1. Pick a random ID
    skull_id = random.choice(skull_ids)
    img_path = os.path.join(skull_root_dir, "images", f"sample_{skull_id}.npy")
    lbl_path = os.path.join(skull_root_dir, "labels", f"sample_{skull_id}.npy")
    skull_img = np.load(img_path)
    skull_lbl = np.load(lbl_path)
    if skull_img is None or skull_lbl is None:
        return None, None
    # 2. Augmentations
    k_rot = random.randint(0, 3)
    axes = random.choice([(0, 1), (0, 2), (1, 2)])
    skull_img = np.rot90(skull_img, k=k_rot, axes=axes)
    skull_lbl = np.rot90(skull_lbl, k=k_rot, axes=axes)
    if random.random() > 0.5:
        axis = random.randint(0, 2)
        skull_img = np.flip(skull_img, axis=axis)
        skull_lbl = np.flip(skull_lbl, axis=axis)
    return skull_img, skull_lbl


def generate_elastic_indices(shape, alpha=15, sigma=3):
    dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = (np.reshape(x + dx, (-1, 1)),
               np.reshape(y + dy, (-1, 1)),
               np.reshape(z + dz, (-1, 1)))
    return indices


def apply_elastic_indices(image, indices, order=1):
    shape = image.shape
    return ndimage.map_coordinates(image, indices, order=order, mode='reflect').reshape(shape)


def fuse_vessel_and_skull(v_img, v_mask, bg_img, s_img, s_lbl):
    final_img = v_img.copy()
    final_mask = v_mask.copy()
    # 1. Soft Masking
    mask_replace_binary = (s_lbl == 0) | (s_lbl == 1)
    mask_float = mask_replace_binary.astype(np.float32)
    soft_mask = ndimage.gaussian_filter(mask_float, sigma=0.8)

    # 2.  Blend background into the vessel image
    final_img = (final_img * (1.0 - soft_mask)) + (bg_img * soft_mask)
    final_mask[soft_mask > 0.5] = 0

    # 3. skull structure
    mask_shell = (s_lbl == 1)
    if mask_shell.any():
        skull_texture = s_img
        skull_texture = skull_texture * 0.85
        # maximum blending inside the skull shell region
        current_pixels = final_img[mask_shell]
        skull_pixels = skull_texture[mask_shell]
        fused_pixels = np.maximum(current_pixels, skull_pixels)
        final_img[mask_shell] = fused_pixels
    final_img = np.clip(final_img, 0, 1.0)
    final_mask = final_mask.astype(np.uint8)
    return final_img, final_mask

def main(config):
    skull_dir = config["skull_dir"]
    vessel_dir = config["vessel_dir"]
    background_dir = config["background_dir"]
    final_output = config["final_output"]
    n_final_outputs = config["n_final_outputs"]

    vessel_ids = [d for d in os.listdir(vessel_dir) if os.path.isdir(os.path.join(vessel_dir, d))]
    bg_ids = [d for d in os.listdir(background_dir) if os.path.isdir(os.path.join(background_dir, d))]
    skull_img_dir = os.path.join(skull_dir, "images")
    skull_files = glob.glob(os.path.join(skull_img_dir, "*.npy"))
    skull_ids = [os.path.basename(f).replace('sample_', '').replace('.npy', '') for f in skull_files]
    if len(vessel_ids) == 0 or len(bg_ids) == 0 or len(skull_ids) == 0:
        print("Error: One or more input directories appear empty or incorrect structure.")
        return
    print(f"Found {len(vessel_ids)} vessels, {len(bg_ids)} backgrounds, {len(skull_ids)} skulls.")

    selected_vessel_ids = random.choices(vessel_ids, k=n_final_outputs)
    count = 0

    for v_id in selected_vessel_ids:
        # Load Vessel
        v_img_path = os.path.join(vessel_dir, v_id, "img.npy")
        v_mask_path = os.path.join(vessel_dir, v_id, "mask.npy")  # Assuming mask name
        v_img = np.load(v_img_path)
        v_mask = np.load(v_mask_path)

        # Load Random Background
        bg_id = random.choice(bg_ids)
        bg_img_path = os.path.join(background_dir, bg_id, "img.npy")
        bg_img = np.load(bg_img_path)

        # Load Random Skull
        s_img, s_lbl = get_random_skull(skull_dir, skull_ids)

        if any(x is None for x in [v_img, v_mask, bg_img, s_img, s_lbl]):
            print(f"Skipping ID {v_id} due to loading error.")
            continue
        if v_img.shape != s_img.shape:
            print(f"Shape mismatch: Vessel {v_img.shape} vs Skull {s_img.shape}. Skipping.")
            continue

        # Execute Injection
        fused_img, fused_mask = fuse_vessel_and_skull(v_img, v_mask, bg_img, s_img, s_lbl)

        # Save
        save_path = os.path.join(final_output, str(count))
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "img.npy"), fused_img)
        np.save(os.path.join(save_path, "mask.npy"), fused_mask)

        count += 1
    print(f"Done. {count} images saved to {final_output}")

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse vessels with skulls.")
    parser.add_argument("--config", type=str, default="skull_config.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    os.makedirs(config["final_output"], exist_ok=True)
    main(config)