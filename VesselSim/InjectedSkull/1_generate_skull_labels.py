import argparse
import os
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter


class RealisticPhantomGenerator:
    def __init__(self, shape=(96, 96, 96)):
        self.shape = shape
        z, y, x = np.indices(shape)
        self.coords = np.stack([z, y, x])

    def generate_shell_geometry(self, thickness=3):
        # 1. Define Ellipsoid
        radii = np.random.uniform(100, 300, 3) ** 2
        patch_center = np.array(self.shape) / 2.0
        direction = np.random.normal(0, 1, 3)
        direction /= np.linalg.norm(direction)
        avg_radius = np.sqrt(np.mean(radii))
        shift = np.random.uniform(-40, 40)
        ellipsoid_center = patch_center + direction * (avg_radius + shift)

        # 2. Distance Field
        coords_shifted = self.coords - ellipsoid_center[:, None, None, None]
        term_z = (coords_shifted[0] ** 2) / radii[0]
        term_y = (coords_shifted[1] ** 2) / radii[1]
        term_x = (coords_shifted[2] ** 2) / radii[2]
        dist_sq = term_z + term_y + term_x

        # 3. Boundaries
        shell_width = thickness / avg_radius * 2.0
        offset_type = np.random.choice(['in', 'out', 'center'])
        if offset_type == 'in':
            low, high = 1.0 - shell_width, 1.0
        elif offset_type == 'out':
            low, high = 1.0, 1.0 + shell_width
        else:
            half = shell_width / 2
            low, high = 1.0 - half, 1.0 + half

        # 4. Labels
        labels = np.zeros(self.shape, dtype=np.uint8)
        labels[dist_sq < low] = 2  # Brain
        skull_mask = (dist_sq >= low) & (dist_sq <= high)
        labels[skull_mask] = 1  # Skull

        return labels, dist_sq, (low, high)

    def _texture_spongy(self, wall_pos, shape): # Bright edges, noisy/porous center.
        # Random noise blobs
        noise = np.random.normal(0.5, 0.2, shape)
        texture_noise = gaussian_filter(noise, sigma=0.5)
        texture_noise = (texture_noise - texture_noise.min()) / (np.ptp(texture_noise) + 1e-6)
        # Profile: How solid are the walls? ->  Power < 1.0 makes the solid walls thicker
        profile_curve = wall_pos ** np.random.uniform(0.5, 1.5)
        cortical_val = np.random.uniform(0.9, 1.0)
        marrow_val = np.random.uniform(0.3, 0.6)
        # Blend
        base_intensity = marrow_val + (cortical_val - marrow_val) * profile_curve
        texture_factor = 1.0 - profile_curve
        return base_intensity * (1 - 0.3 * texture_factor * texture_noise)

    def _texture_layered(self, dist_map, bounds, shape): # Wiggly, sharp laminar layers with internal warping.
        low, high = bounds
        mid_point = (low + high) / 2.0
        shell_width = high - low
        # 1. Base normalized pos (0 to 1 across the shell)
        norm_pos = (dist_map - low) / (shell_width + 1e-6)
        # 2. create warping/turbulence
        noise = np.random.normal(0, 1, shape)
        turbulence = gaussian_filter(noise, sigma=3.0)
        turbulence = (turbulence - turbulence.mean()) / (turbulence.std() + 1e-6)
        warp_strength = 0.15
        warped_pos = norm_pos + (turbulence * warp_strength)
        # 3. wave pattern/layers
        num_layers = np.random.uniform(4, 7)
        striation_pattern = np.cos(warped_pos * num_layers * 2 * np.pi)
        striation_pattern = np.clip(striation_pattern * 1.5, -1, 1)
        striation_pattern = (striation_pattern + 1) / 2.0  # 0.0 to 1.0 range
        # 4. intensities
        bone_bright = np.random.uniform(0.85, 1.0)
        bone_dark = np.random.uniform(0.2, 0.4)
        final_texture = bone_dark + (bone_bright - bone_dark) * striation_pattern
        # 5. smooth boundary -> skull shape stays smooth, even if the insides are crazy.
        wall_dist_norm = np.abs(dist_map - mid_point) / (shell_width / 2.0)
        wall_dist_norm = np.clip(wall_dist_norm, 0, 1)
        wall_dist_norm = wall_dist_norm ** 6
        if np.random.rand() > 0.5:
            final_texture = np.maximum(final_texture, wall_dist_norm * bone_bright)
        else:
            final_texture = np.maximum(final_texture, wall_dist_norm * bone_dark)
        return final_texture

    def get_random_skull(self):
        # 1. geometry
        thickness = np.random.randint(5, 16)
        labels, dist_map, bounds = self.generate_shell_geometry(thickness)
        skull_mask = (labels == 1)
        # 2. texture (Spongy vs Layered)
        if np.random.rand() > 0.5:# Spongy
            low, high = bounds
            mid_point = (low + high) / 2.0
            half_width = (high - low) / 2.0
            # Calc normalized position (0=center, 1=edge)
            wall_pos = np.abs(dist_map - mid_point) / (half_width + 1e-6)
            wall_pos = np.clip(wall_pos, 0, 1)
            final_texture = self._texture_spongy(wall_pos, self.shape)
            mode_name = "Spongy"
        else: # layers
            final_texture = self._texture_layered(dist_map, bounds, self.shape)
            mode_name = "Layered"

        # 3. apply to volume
        final_vol = np.zeros(self.shape, dtype=np.float32)
        final_vol[skull_mask] = final_texture[skull_mask]
        final_vol = gaussian_filter(final_vol, sigma=0.6)
        final_vol = np.clip(final_vol, 0.0, 1.0)
        return final_vol, labels, mode_name

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def generate_dataset(n_samples, output_dir="skull_labels"):
    gen = RealisticPhantomGenerator()
    vol_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(vol_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    print(f"Starting generation of {n_samples} samples...")
    print(f"Output directory: {output_dir}")

    count = 0
    max_retries = 20

    while count < n_samples:
        # Retry loop for empty volumes
        valid_sample = False
        for _ in range(max_retries):
            vol, lbl, mode = gen.get_random_skull()
            # Check if we have enough bone (>2% volume)
            if (lbl == 1).mean() > 0.02:
                valid_sample = True
                break

        if not valid_sample:
            print(f"Warning: Failed to generate valid sample for index {count}. Skipping...")
            continue

        filename = f"sample_{count:04d}.npy"
        np.save(os.path.join(vol_dir, filename), vol)
        np.save(os.path.join(lbl_dir, filename), lbl)
        count += 1

    print("Done!")


def main():
    parser = argparse.ArgumentParser( description="Generate synthetic skull phantom images and labels from a YAML config." )
    parser.add_argument("--config",type=str, default="skull_config.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    n_samples = config["n_skull_samples"]
    output_dir = config["skull_dir"]
    generate_dataset(n_samples=n_samples, output_dir=output_dir)


if __name__ == "__main__":
    main()
