import argparse
import random
from scipy.ndimage import label, center_of_mass, zoom
import os
import glob
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_largest_component_ratio(patch):
    if np.sum(patch) == 0: return 0.0
    labeled_array, num_features = label(patch)
    if num_features == 0: return 0.0
    component_sizes = np.bincount(labeled_array.ravel())[1:]
    if len(component_sizes) == 0: return 0.0
    return component_sizes.max() / component_sizes.sum()

def evaluate_patch(patch):
    # --- CRITERIA 1: Volume Threshold ---
    density = np.mean(patch)
    if density < 0.005: return -1000  # Immediate fail

    # --- CRITERIA 2: Face Contact ---
    faces = [
        patch[0, :, :], patch[-1, :, :],
        patch[:, 0, :], patch[:, -1, :],
        patch[:, :, 0], patch[:, :, -1]
    ]
    faces_touched = sum(1 for f in faces if np.sum(f) > 0)

    # Bonus for touching opposite faces (indicates flow-through)
    x_flow = (np.sum(faces[0]) > 0 and np.sum(faces[1]) > 0)
    y_flow = (np.sum(faces[2]) > 0 and np.sum(faces[3]) > 0)
    z_flow = (np.sum(faces[4]) > 0 and np.sum(faces[5]) > 0)
    flow_bonus = sum([x_flow, y_flow, z_flow])

    # --- CRITERIA 3: Connectivity (No Islands) ---
    connectivity_ratio = get_largest_component_ratio(patch)

    score = (connectivity_ratio * 100) + (faces_touched * 30) + (flow_bonus * 50) + (density * 5000)
    return score

def smart_zoom_repair(patch, target_size=96, min_density=0.01):
    density = np.mean(patch)
    faces = [
        patch[0, :, :], patch[-1, :, :],
        patch[:, 0, :], patch[:, -1, :],
        patch[:, :, 0], patch[:, :, -1]
    ]
    touched = sum(1 for f in faces if np.sum(f) > 0)

    if density >= min_density and touched > 0:
        return patch, False

    coords = np.argwhere(patch > 0)
    if coords.size == 0: return patch, False  # Empty, can't zoom

    min_z, min_y, min_x = coords.min(axis=0)
    max_z, max_y, max_x = coords.max(axis=0) + 1

    d_len = max_z - min_z
    h_len = max_y - min_y
    w_len = max_x - min_x

    longest_side = max(d_len, h_len, w_len)
    cube_vol = np.zeros((longest_side, longest_side, longest_side), dtype=patch.dtype)
    z_offset = (longest_side - d_len) // 2
    y_offset = (longest_side - h_len) // 2
    x_offset = (longest_side - w_len) // 2

    tight_crop = patch[min_z:max_z, min_y:max_y, min_x:max_x]

    cube_vol[z_offset:z_offset + d_len,
    y_offset:y_offset + h_len,
    x_offset:x_offset + w_len] = tight_crop

    zoom_factor = target_size / longest_side

    cube_vol_float = cube_vol.astype(float)
    zoomed_patch_float = zoom(cube_vol_float, zoom_factor, order=1)

    zoomed_patch = (zoomed_patch_float > 0.5).astype(np.uint8)

    final_patch = np.zeros((target_size, target_size, target_size), dtype=np.uint8)

    zd, zh, zw = zoomed_patch.shape
    d_off = max(0, (zd - target_size) // 2)
    h_off = max(0, (zh - target_size) // 2)
    w_off = max(0, (zw - target_size) // 2)

    d_len = min(target_size, zd)
    h_len = min(target_size, zh)
    w_len = min(target_size, zw)

    final_patch[:d_len, :h_len, :w_len] = zoomed_patch[d_off:d_off + d_len, h_off:h_off + h_len, w_off:w_off + w_len]

    return final_patch, True


def iterative_zoom_rescue(patch, min_req=0.01, target_size=96):
    current_patch = patch.copy()
    crop_steps = range(85, 40, -10)

    for sub_size in crop_steps:
        # 1. Find Center of Mass
        try:
            cm = center_of_mass(current_patch)
            if np.any(np.isnan(cm)): cm = [target_size // 2] * 3
        except:
            cm = [target_size // 2] * 3
        # 2. Define Crop Box around Center
        cx, cy, cz = int(cm[0]), int(cm[1]), int(cm[2])
        half = sub_size // 2
        # Clip to bounds
        z1 = np.clip(cx - half, 0, target_size - sub_size)
        y1 = np.clip(cy - half, 0, target_size - sub_size)
        x1 = np.clip(cz - half, 0, target_size - sub_size)
        # 3. Extract Sub-volume
        sub_vol = current_patch[z1:z1 + sub_size, y1:y1 + sub_size, x1:x1 + sub_size]
        # 4. Zoom back up to 96
        zoom_factor = target_size / sub_size
        sub_vol_float = sub_vol.astype(float)
        zoomed_float = zoom(sub_vol_float, zoom_factor, order=1)
        zoomed_patch = (zoomed_float > 0.5).astype(np.uint8)
        # 5. Fix Dimensions (Pad/Crop to exact 96)
        final_attempt = np.zeros((target_size, target_size, target_size), dtype=np.uint8)
        zd, zh, zw = zoomed_patch.shape
        d_len = min(target_size, zd)
        h_len = min(target_size, zh)
        w_len = min(target_size, zw)
        final_attempt[:d_len, :h_len, :w_len] = zoomed_patch[:d_len, :h_len, :w_len]
        # 6. Check Density
        if np.mean(final_attempt) >= min_req:
            return final_attempt, True
    return patch, False

def extract_best_patch(volume, crop_size=96, max_attempts=50):
    vol_x, vol_y, vol_z = volume.shape
    best_patch = None
    best_score = -float('inf')

    # Attempt 1: Center of Mass
    try:
        if np.sum(volume) > 0:
            cm = center_of_mass(volume)
            start_x = np.clip(int(cm[0] - crop_size / 2), 0, vol_x - crop_size)
            start_y = np.clip(int(cm[1] - crop_size / 2), 0, vol_y - crop_size)
            start_z = np.clip(int(cm[2] - crop_size / 2), 0, vol_z - crop_size)

            patch = volume[start_x:start_x + crop_size, start_y:start_y + crop_size, start_z:start_z + crop_size]
            score = evaluate_patch(patch)
            if score > best_score:
                best_score = score
                best_patch = patch
    except:
        pass

    # Attempt 2: Random Sliding Window
    for i in range(max_attempts):
        x0 = random.randint(0, vol_x - crop_size)
        y0 = random.randint(0, vol_y - crop_size)
        z0 = random.randint(0, vol_z - crop_size)

        current_patch = volume[x0:x0 + crop_size, y0:y0 + crop_size, z0:z0 + crop_size]
        current_score = evaluate_patch(current_patch)

        if current_score > best_score:
            best_score = current_score
            best_patch = current_patch

    # Fallback if everything failed
    if best_patch is None:
        mid_x, mid_y, mid_z = vol_x // 2, vol_y // 2, vol_z // 2
        half = crop_size // 2
        best_patch = volume[mid_x - half:mid_x + half, mid_y - half:mid_y + half, mid_z - half:mid_z + half]

    # Check if we need to zoom to fix density or faces
    final_patch, was_repaired = smart_zoom_repair(best_patch, target_size=crop_size, min_density=0.01)
    final_score = evaluate_patch(final_patch)
    was_zoomed = False
    if final_score <= -1000:
        final_patch, was_zoomed = iterative_zoom_rescue(final_patch, min_req=0.01)
        if was_zoomed:
            final_score = evaluate_patch(final_patch)

    return final_patch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--config", type=str, default="generation_config.yaml", help="Path to config YAML file", )
    args = parser.parse_args()
    cfg = load_config(args.config)

    in_dir = cfg["builder_output_dir"]
    out_dir = cfg["final_patches_dir"]
    patch_size = int(cfg["final_patch_size"])
    max_attempts = int(cfg["patch_extraction_attempts"])
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(in_dir, "*.npy")))
    if not paths: raise FileNotFoundError(f"No .npy files found in {in_dir}")

    print("--- Processing Started ---")
    print(f"Patch size: {patch_size}")
    print(f"Found {len(paths)} .npy files\n")

    current_id = 0

    for pth in paths:
        name = os.path.basename(pth)
        vol = np.load(pth)
        if vol.ndim != 3: raise ValueError(f"Shape {vol.shape} is not 3D")
        if min(vol.shape) < patch_size: raise ValueError(f"Volume {vol.shape} smaller than patch {patch_size}")
        final_patch = extract_best_patch(vol, crop_size=patch_size, max_attempts=50)
        id_folder = os.path.join(out_dir, str(current_id))
        os.makedirs(id_folder, exist_ok=True)
        out_name = f"mask_{name}"
        out_path = os.path.join(id_folder, out_name)
        np.save(out_path, final_patch)
        print(f"Saved: {out_name}")
        current_id += 1
    print("Done.")