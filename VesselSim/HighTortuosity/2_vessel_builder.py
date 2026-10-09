import os
import json
import re
from pathlib import Path
import yaml
import numpy as np
import random
import math

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class VesselBuilder:
    def __init__(self, config, volume_size):
        self.config = config
        self.volume_size = volume_size
        self.volume = np.zeros(volume_size, dtype=np.uint8)
        self.root = config['root_geometry']
        self.branching = config['branching_model']
        self.radius = config['radius_model']
        self.tortuosity = config['tortuosity_model']
        self.total_segments = 0
        self.root_position = None
        self.segments = []
        self._bifurcated = set()
        self._drop_reasons = {}

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else np.array([1.0, 0.0, 0.0])

    def _get_random_orthogonal(self, v):
        random_vec = np.random.randn(3)
        ortho = np.cross(v, random_vec)
        if np.linalg.norm(ortho) < 1e-6:
            ortho = np.cross(v, np.array([1.0, 0.0, 0.0]))
        return self._normalize(ortho)

    def _rotate_vector(self, vector, axis, angle_deg):
        angle_rad = np.radians(angle_deg)
        vector = self._normalize(vector)
        axis = self._normalize(axis)
        return (vector * math.cos(angle_rad) +
                np.cross(axis, vector) * math.sin(angle_rad) +
                axis * np.dot(axis, vector) * (1 - math.cos(angle_rad)))

    def _get_sphere_mask_local(self, center_local, shape, radius):
        grid_x, grid_y, grid_z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        dist_sq = ((grid_x - center_local[0]) ** 2 +
                   (grid_y - center_local[1]) ** 2 +
                   (grid_z - center_local[2]) ** 2)
        return dist_sq <= radius ** 2

    def _check_collision_volumetric(self, curr_pos, next_pos, radius):
        r = int(math.ceil(radius))
        x_min = max(0, int(min(curr_pos[0], next_pos[0]) - r))
        x_max = min(self.volume_size[0], int(max(curr_pos[0], next_pos[0]) + r + 1))
        y_min = max(0, int(min(curr_pos[1], next_pos[1]) - r))
        y_max = min(self.volume_size[1], int(max(curr_pos[1], next_pos[1]) + r + 1))
        z_min = max(0, int(min(curr_pos[2], next_pos[2]) - r))
        z_max = min(self.volume_size[2], int(max(curr_pos[2], next_pos[2]) + r + 1))

        if x_min >= x_max or y_min >= y_max or z_min >= z_max:
            return True
        sub_volume = self.volume[x_min:x_max, y_min:y_max, z_min:z_max]
        curr_local = curr_pos - np.array([x_min, y_min, z_min])
        next_local = next_pos - np.array([x_min, y_min, z_min])
        box_shape = sub_volume.shape

        mask_curr = self._get_sphere_mask_local(curr_local, box_shape, radius+4)
        mask_next = self._get_sphere_mask_local(next_local, box_shape, radius+3)
        mask_check = mask_next & (~mask_curr)

        if np.any(sub_volume[mask_check] > 0):
            return True
        return False

    def _draw_sphere(self, position, radius):
        x, y, z = position
        r = int(math.ceil(radius))
        x_min, x_max = max(0, int(x - r)), min(self.volume_size[0], int(x + r + 1))
        y_min, y_max = max(0, int(y - r)), min(self.volume_size[1], int(y + r + 1))
        z_min, z_max = max(0, int(z - r)), min(self.volume_size[2], int(z + r + 1))

        if x_min >= x_max or y_min >= y_max or z_min >= z_max: return

        grid_x, grid_y, grid_z = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
        dist_sq = (grid_x - x) ** 2 + (grid_y - y) ** 2 + (grid_z - z) ** 2
        mask = dist_sq <= radius ** 2
        self.volume[x_min:x_max, y_min:y_max, z_min:z_max][mask] = 1

    def _drop(self, parent_id, reason):
        self._drop_reasons.setdefault(parent_id, []).append(reason)

    def build(self):
        start_pos = np.array(self.root['origin_voxel'], dtype=float)
        init_dir = np.array(self.root.get('initial_direction', [0, 0, 1]), dtype=float)
        if np.linalg.norm(init_dir) == 0: init_dir = np.array([0, 0, 1])
        start_dir = self._normalize(init_dir)
        start_radius = self.root['initial_radius']
        self.root_position = start_pos.tolist()

        self._draw_sphere(start_pos, start_radius)

        queue = [(start_pos, start_dir, start_radius, 0, None)]
        max_segments = self.branching['max_total_segments']

        while queue and  self.total_segments < max_segments:
            curr_pos, curr_dir, curr_radius, depth, parent_id = queue.pop(0)

            if depth > int(self.branching["max_depth"]):
                self._drop(parent_id, "max_depth")
                continue
            if curr_radius < float(self.radius["min_radius_vox"]):
                self._drop(parent_id, "min_radius")
                continue

            seg_len = random.randint(self.branching['segment_length_vox']['min'], self.branching['segment_length_vox']['max'])
            steps = int(seg_len)
            segment_retries = 6
            segment_succeeded = False

            seg_start_pos = curr_pos.copy()
            seg_start_dir = curr_dir.copy()
            bend_axis_base = self._get_random_orthogonal(seg_start_dir)
            max_turn_deg = max(0.0, float(self.tortuosity['segment_max_curvature']))
            turn_deg_base = random.uniform(-max_turn_deg, max_turn_deg)
            angle_jitter_scale = self.tortuosity['collision_jitter_scale']
            axis_jitter_angle = 5.0

            for attempt in range(segment_retries):
                test_pos = seg_start_pos.copy()
                test_dir = seg_start_dir.copy()

                turn_jitter = random.uniform(-max_turn_deg * angle_jitter_scale, max_turn_deg * angle_jitter_scale)
                turn_deg = turn_deg_base + turn_jitter

                bend_axis = bend_axis_base
                if axis_jitter_angle > 0:
                    wobble_axis = self._get_random_orthogonal(bend_axis_base)
                    bend_axis = self._rotate_vector(bend_axis_base, wobble_axis, random.uniform(-axis_jitter_angle, axis_jitter_angle))
                    bend_axis = self._normalize(bend_axis)

                path_positions = []
                path_dirs = []
                blocked = False

                for _ in range(steps):
                    next_dir = self._rotate_vector(test_dir, bend_axis, turn_deg)
                    next_dir = self._normalize(next_dir)
                    next_pos = test_pos + next_dir

                    if self._check_collision_volumetric(test_pos, next_pos, curr_radius):
                        blocked = True
                        break

                    path_positions.append(next_pos)
                    path_dirs.append(next_dir)
                    test_pos = next_pos
                    test_dir = next_dir

                if not blocked:
                    for p in path_positions:
                        self._draw_sphere(p, curr_radius)

                    curr_pos = test_pos
                    curr_dir = test_dir
                    segment_succeeded = True
                    break

            if not segment_succeeded:
                self._drop(parent_id, "segment_blocked")
                continue
            self.total_segments += 1

            seg_id = len(self.segments)
            self.segments.append({
                "segment_id": seg_id,
                "parent_segment_id": parent_id,
                "depth": depth,
                "radius": float(curr_radius),
                "start_position": seg_start_pos.tolist(),
                "path_positions": [p.tolist() for p in path_positions],
            })

            base_bif_prob = self.branching['bifurcation_probability']
            decay = self.branching['bifurcation_depth_decay']
            bif_prob = base_bif_prob * (decay ** depth)

            if random.random() < bif_prob:
                self._bifurcated.add(seg_id)
                angle = self.branching['bifurcation_angle']
                rot_axis = self._get_random_orthogonal(curr_dir)

                dir_a = self._rotate_vector(curr_dir, rot_axis, angle)
                dir_b = self._rotate_vector(curr_dir, rot_axis, -angle)
                new_rad = curr_radius * self.radius['radius_decay_factor']

                pos_a = curr_pos + dir_a
                pos_b = curr_pos + dir_b
                if not self._check_collision_volumetric(curr_pos, pos_a, new_rad):
                    queue.append((curr_pos, dir_a, new_rad, depth + 1, seg_id))
                else:
                    self._drop(seg_id, "bifurcation_blocked")
                if not self._check_collision_volumetric(curr_pos, pos_b, new_rad):
                    queue.append((curr_pos, dir_b, new_rad, depth + 1, seg_id))
                else:
                    self._drop(seg_id, "bifurcation_blocked")


            else:
                term_factor = self.branching['termination_factor']
                term_prob = 1.0 - math.exp(-term_factor*depth)

                if random.random() > term_prob:
                    pos_next = curr_pos + curr_dir
                    if not self._check_collision_volumetric(curr_pos, pos_next, curr_radius):
                        queue.append((curr_pos, curr_dir, curr_radius, depth, seg_id))
                    else:
                        self._drop(seg_id, "continuation_blocked")
                else:
                    self._drop(seg_id, "termination_probability")

        for entry in queue:
            self._drop(entry[4], "max_segments")

        return self.volume

    def raw_tree(self):
        return {"root_position": self.root_position, "segments": self.segments}

    def canonical_graph(self):
        # Nodes: root, bifurcations (with >= 1 accepted child) and terminal endpoints.
        # Edges: chains of continuation segments between nodes, with exact centerline points.
        children = {}
        for seg in self.segments:
            children.setdefault(seg["parent_segment_id"], []).append(seg["segment_id"])

        root = {"id": 0, "type": "root", "position": self.root_position}
        if None not in children:
            root["termination_reasons"] = self._drop_reasons.get(None, [])
        nodes = [root]
        edges = []

        pending = [(0, seg_id) for seg_id in children.get(None, [])]
        while pending:
            source, seg_id = pending.pop(0)
            first = self.segments[seg_id]
            points = [first["start_position"]]
            radii = [first["radius"]]
            depths = [first["depth"]]
            segment_ids = []

            while True:
                seg = self.segments[seg_id]
                n = len(seg["path_positions"])
                segment_ids.append(seg_id)
                points += seg["path_positions"]
                radii += [seg["radius"]] * n
                depths += [seg["depth"]] * n
                kids = children.get(seg_id, [])
                if seg_id in self._bifurcated or len(kids) != 1:
                    break
                seg_id = kids[0]

            node = {"id": len(nodes), "position": points[-1]}
            if kids:
                node["type"] = "bifurcation"
                node["n_children"] = len(kids)
                pending += [(node["id"], kid) for kid in kids]
            else:
                node["type"] = "terminal"
                node["termination_reasons"] = self._drop_reasons.get(seg_id, [])
            nodes.append(node)

            edges.append({
                "id": len(edges),
                "source": source,
                "target": node["id"],
                "segment_ids": segment_ids,
                "points": points,
                "radii": radii,
                "depths": depths,
            })

        return {"nodes": nodes, "edges": edges}

def next_attempt_for_base(output_dir: str, base: str) -> int:
    if not os.path.isdir(output_dir):
        return 0
    pat = re.compile(rf"^{re.escape(base)}_(\d{{2}})$")
    max_att = -1
    for fn in os.listdir(output_dir):
        m = pat.match(fn)
        if m and os.path.isdir(os.path.join(output_dir, fn)):
            max_att = max(max_att, int(m.group(1)))
    return max_att + 1

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    cfg = load_config("generation_config.yaml")

    volume_size = tuple(cfg["initial_volume_size"])
    parameters_dir = Path(cfg["parameters_dir"])
    output_dir = Path(cfg["builder_output_dir"])
    masks_per_json = int(cfg["masks_per_json"])

    json_paths = sorted(parameters_dir.glob("*.json"))
    if len(json_paths) == 0:
        print(f"No parameter files found in: {parameters_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(json_paths)} JSON parameter files.")
    print(f"Generating {masks_per_json} vessel masks per JSON.")
    print(f"Saving masks and graphs to: {output_dir}")

    for json_path in json_paths:
        base = json_path.stem
        print(f"Processing {json_path.name}")
        params = load_json(json_path)

        for _ in range(masks_per_json):
            try:
                builder = VesselBuilder(params, volume_size)
                mask = builder.build()
                attempt = next_attempt_for_base(output_dir, base)
                sample_dir = output_dir / f"{base}_{attempt:02d}"
                sample_dir.mkdir()
                metadata = {
                    "parameters_file": json_path.name,
                    "volume_shape": list(volume_size),
                    "coordinates": "continuous voxel indices (x, y, z) into vessel_mask.npy, i.e. vessel_mask[x, y, z]",
                    "units": "voxels",
                }
                np.save(sample_dir / "vessel_mask.npy", mask)
                save_json(sample_dir / "raw_tree.json", {"metadata": metadata, **builder.raw_tree()})
                save_json(sample_dir / "canonical_graph.json", {"metadata": metadata, **builder.canonical_graph()})
            except Exception as e:
                print(f"  Failed for {json_path.name}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()