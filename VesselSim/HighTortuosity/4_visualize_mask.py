import argparse
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw


def load_npy(path):
    data = np.load(path)
    if data.ndim == 4 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {data.shape}")
    return data


def make_grid(data, spacing):
    nx, ny, nz = data.shape
    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.point_data["values"] = np.asfortranarray(
        data.astype(np.float32)
    ).ravel(order="F")
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


def surface_from_binary_volume(mask, spacing):
    grid = make_grid(mask.astype(np.float32), spacing)
    surf = grid.contour(isosurfaces=[0.5], scalars="values")
    return surf, grid

def render_mask_cube(mask, spacing, camera_position, screenshot_size=(400, 400)):
    surf, grid = surface_from_binary_volume(mask, spacing)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", type=str, required=True, help="Path to .npy file to visualize",)
    parser.add_argument("--out", type=str, default="mask_visualization.png", help="Output PNG path",)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), help="Voxel spacing, default is 1 1 1",)
    parser.add_argument("--size", type=int, nargs=2, default=(600, 600),help="Screenshot size, default is 600 600",)
    args = parser.parse_args()

    mask = load_npy(args.mask)
    mask = mask > 0
    spacing = tuple(args.spacing)
    grid = make_grid(mask.astype(np.float32), spacing)
    camera_position = get_camera(grid)

    mask_cube = render_mask_cube(mask, spacing, camera_position,  screenshot_size=tuple(args.size),)
    mask_cube.save(args.out)

if __name__ == "__main__":
    main()