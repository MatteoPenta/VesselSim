import argparse
import os.path

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw

def load_npy(path):
    data = np.load(path).astype(np.float32)
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.shape != (96, 96, 96):
        raise ValueError(f"Expected shape (96, 96, 96), got {data.shape}")
    return data

def normalize01(x, pmin=1, pmax=99):
    lo, hi = np.percentile(x, [pmin, pmax])
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)

def make_grid(data, spacing):
    nx, ny, nz = data.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.point_data["values"] = np.asfortranarray(data.astype(np.float32)).ravel(order="F")
    return grid

def get_camera(grid):
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin
    d = max(dx, dy, dz)
    return [
        (cx + 2.6 * d, cy - 2.6 * d, cz + 2.0 * d),
        (cx, cy, cz),
        (0, 0, 1),
    ]

def render_image_cube(img_patch, spacing, camera_position, screenshot_size=(600, 600)):
    img_patch = normalize01(img_patch)
    grid = make_grid(img_patch, spacing)
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

    sx = grid.slice(
        normal="x",
        origin=(xmax - 0.01, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)),
    )
    sy = grid.slice(
        normal="y",
        origin=(0.5 * (xmin + xmax), ymin + 0.01, 0.5 * (zmin + zmax)),
    )
    sz = grid.slice(
        normal="z",
        origin=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), zmax - 0.01),
    )
    outline = grid.outline()
    p = pv.Plotter(off_screen=True, window_size=screenshot_size)
    p.set_background("white")
    for s in [sx, sy, sz]:
        if s.n_points == 0:
            continue
        p.add_mesh(
            s,
            scalars="values",
            cmap="gray",
            clim=(0, 1),
            show_scalar_bar=False,
            lighting=False,
        )
    p.add_mesh(outline, color="black", line_width=2, opacity=0.5)
    p.camera_position = camera_position
    p.camera.zoom(1.25)
    img = p.screenshot(return_img=True)
    p.close()
    return Image.fromarray(img)

def surface_from_binary_volume(data, spacing):
    grid = make_grid(data.astype(np.float32), spacing)
    surf = grid.contour(isosurfaces=[0.5], scalars="values")
    return surf, grid

def render_mask_cube(mask_patch, spacing, camera_position, screenshot_size=(600, 600)):
    surf, grid = surface_from_binary_volume(mask_patch, spacing)

    if surf.n_points == 0:
        img = Image.new("RGB", screenshot_size, "white")
        draw = ImageDraw.Draw(img)
        draw.line((0, 0, img.size[0], img.size[1]), fill=(200, 200, 200), width=5)
        draw.line((0, img.size[1], img.size[0], 0), fill=(200, 200, 200), width=5)
        return img
    outline = grid.outline()
    p = pv.Plotter(off_screen=True, window_size=screenshot_size)
    p.set_background("white")
    p.add_mesh(
        surf,
        color="#d62728",
        smooth_shading=False,
        specular=0.0,
        metallic=0.0,
        roughness=1.0,
        ambient=0.3,
        diffuse=0.8,
        show_edges=False,
    )
    p.add_mesh(outline, color="black", line_width=2, opacity=0.5)
    p.camera_position = camera_position
    p.camera.zoom(1.25)
    img = p.screenshot(return_img=True)
    p.close()
    return Image.fromarray(img)

def save_combined(image_cube, mask_cube, out_path):
    w, h = image_cube.size
    canvas = Image.new("RGB", (2 * w, h), "white")
    canvas.paste(image_cube.convert("RGB"), (0, 0))
    canvas.paste(mask_cube.convert("RGB"), (w, 0))
    canvas.save(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume", type=str, required=True, help="Path to folder id to visualize", )
    parser.add_argument("--out", type=str, default="skull_visualization.png", help="Output PNG path", )
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()

    image_path = os.path.join(args.volume, "img.npy")
    mask_path = os.path.join(args.volume, "mask.npy")
    image = load_npy(image_path)
    mask = load_npy(mask_path) > 0
    if image.shape != mask.shape:
        raise ValueError(f"Image and mask shapes differ: {image.shape} vs {mask.shape}")

    spacing = tuple(args.spacing)
    grid = make_grid(np.zeros_like(image, dtype=np.float32), spacing)
    camera_position = get_camera(grid)
    image_cube = render_image_cube(image, spacing, camera_position)
    mask_cube = render_mask_cube(mask, spacing, camera_position)
    save_combined(image_cube, mask_cube, args.out)


if __name__ == "__main__":
    main()