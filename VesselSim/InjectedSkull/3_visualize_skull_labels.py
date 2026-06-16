import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml


def visualize_grid(data_dir="skull_labels/images"):
    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    if len(all_files) == 0:
        print("No .npy files found.")
        return

    num_samples = min(9, len(all_files))
    selected_files = random.sample(all_files, num_samples)

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f"Random Skull Samples (Middle Slice)", fontsize=16)
    axes_flat = axes.flat

    for i, ax in enumerate(axes_flat):
        if i < num_samples:
            filename = selected_files[i]
            file_path = os.path.join(data_dir, filename)
            vol = np.load(file_path)
            mid_idx = vol.shape[0] // 2
            slice_img = vol[mid_idx, :, :]
            ax.imshow(slice_img, cmap='bone', vmin=0, vmax=1)
            ax.set_title(filename, fontsize=9)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def load_yaml_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser( description="Generate synthetic skull phantom images and labels from a YAML config." )
    parser.add_argument("--config",type=str, default="skull_config.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    skull_dir = config["skull_dir"]
    skull_label_dir = os.path.join(skull_dir, "images")
    visualize_grid(skull_label_dir)

if __name__ == "__main__":
    main()
