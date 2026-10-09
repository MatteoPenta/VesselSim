import argparse
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
import pyvista as pv
import yaml


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def list_samples(data_dir):
    return sorted(p.name for p in data_dir.iterdir() if (p / "canonical_graph.json").is_file())


def mask_surface(mask):
    # Grid points sit at voxel indices, so the surface shares the graph's coordinates.
    grid = pv.ImageData(dimensions=mask.shape)
    grid.point_data["values"] = np.asfortranarray(mask.astype(np.float32)).ravel(order="F")
    surf = grid.contour(isosurfaces=[0.5], scalars="values").triangulate()
    if surf.n_points == 0:
        return {"x": [], "y": [], "z": [], "i": [], "j": [], "k": []}
    pts = np.round(surf.points, 3)
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    return {
        "x": pts[:, 0].tolist(), "y": pts[:, 1].tolist(), "z": pts[:, 2].tolist(),
        "i": faces[:, 0].tolist(), "j": faces[:, 1].tolist(), "k": faces[:, 2].tolist(),
    }


@lru_cache(maxsize=16)
def load_sample(sample_dir, mtime):
    sample_dir = Path(sample_dir)
    with open(sample_dir / "canonical_graph.json", "r") as f:
        graph = json.load(f)
    mask = np.load(sample_dir / "vessel_mask.npy") > 0
    payload = {
        "shape": list(mask.shape),
        "mesh": mask_surface(mask),
        "nodes": graph["nodes"],
        "edges": graph["edges"],
    }
    return json.dumps(payload).encode()


def make_handler(data_dir):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif url.path == "/api/samples":
                self._send(200, "application/json", json.dumps(list_samples(data_dir)).encode())
            elif url.path == "/api/sample":
                name = parse_qs(url.query).get("name", [""])[0]
                if name not in list_samples(data_dir):
                    self._send(404, "text/plain", f"Unknown sample: {name}".encode())
                    return
                sample_dir = data_dir / name
                try:
                    mtime = (sample_dir / "canonical_graph.json").stat().st_mtime
                    self._send(200, "application/json", load_sample(str(sample_dir), mtime))
                except Exception as e:
                    self._send(500, "text/plain", f"Failed to load {name}: {e}".encode())
            else:
                self._send(404, "text/plain", b"Not found")

        def log_message(self, format, *args):
            pass

    return Handler


PAGE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Vessel graph viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { margin: 0; font: 13px sans-serif; }
  #bar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; padding: 8px 12px; border-bottom: 1px solid #ddd; }
  #info { color: #555; }
  #plot { width: 100vw; height: calc(100vh - 48px); }
</style>
</head>
<body>
<div id="bar">
  <label>Sample <select id="sample"></select></label>
  <label><input type="checkbox" id="showMask" checked> Mask</label>
  <label>Opacity <input type="range" id="opacity" min="0.05" max="1" step="0.05" value="0.25"></label>
  <label><input type="checkbox" id="showEdges" checked> Edges</label>
  <label>Color by <select id="colorBy">
    <option value="radius">radius</option>
    <option value="depth">depth</option>
    <option value="none">uniform</option>
  </select></label>
  <label><input type="checkbox" id="showNodes" checked> Nodes</label>
  <span id="info"></span>
</div>
<div id="plot"></div>
<script>
const $ = id => document.getElementById(id);
const NODE_COLORS = { root: "black", bifurcation: "#1f77b4", terminal: "#ff7f0e" };
let data = null, current = null;

async function loadSamples() {
  const names = await (await fetch("api/samples")).json();
  if (!names.length) { $("info").textContent = "No samples with canonical_graph.json found."; return; }
  $("sample").innerHTML = names.map(n => `<option>${n}</option>`).join("");
  loadSample(names[0]);
}

async function loadSample(name) {
  $("info").textContent = "Loading " + name + "...";
  const res = await fetch("api/sample?name=" + encodeURIComponent(name));
  if (!res.ok) { $("info").textContent = await res.text(); return; }
  data = await res.json();
  current = name;
  const counts = {};
  data.nodes.forEach(n => counts[n.type] = (counts[n.type] || 0) + 1);
  $("info").textContent = `${data.edges.length} edges, ` +
    Object.entries(counts).map(([t, c]) => `${c} ${t}`).join(", ");
  render();
}

function edgeTrace(colorBy) {
  const x = [], y = [], z = [], c = [], text = [];
  for (const e of data.edges) {
    e.points.forEach((p, i) => {
      x.push(p[0]); y.push(p[1]); z.push(p[2]);
      c.push(colorBy === "depth" ? e.depths[i] : e.radii[i]);
      text.push(`edge ${e.id} (${e.source} → ${e.target})<br>segments ${e.segment_ids.join(", ")}` +
                `<br>radius ${e.radii[i].toFixed(3)}<br>depth ${e.depths[i]}`);
    });
    x.push(null); y.push(null); z.push(null); c.push(c[c.length - 1]); text.push("");
  }
  const line = { width: 6 };
  if (colorBy === "none") line.color = "#2ca02c";
  else Object.assign(line, { color: c, colorscale: "Viridis", showscale: true,
                             colorbar: { title: colorBy, x: 1.0 } });
  return { type: "scatter3d", mode: "lines", name: "edges", x, y, z, line, text, hoverinfo: "text" };
}

function nodeTraces() {
  return Object.keys(NODE_COLORS).map(type => {
    const nodes = data.nodes.filter(n => n.type === type);
    return {
      type: "scatter3d", mode: "markers", name: type,
      x: nodes.map(n => n.position[0]), y: nodes.map(n => n.position[1]), z: nodes.map(n => n.position[2]),
      marker: { size: type === "root" ? 7 : 5, color: NODE_COLORS[type] },
      text: nodes.map(n => {
        let t = `node ${n.id} (${n.type})<br>(${n.position.map(v => v.toFixed(2)).join(", ")})`;
        if (n.n_children !== undefined) t += `<br>children ${n.n_children}`;
        if (n.termination_reasons) t += `<br>${n.termination_reasons.join(", ")}`;
        return t;
      }),
      hoverinfo: "text",
    };
  });
}

function render() {
  if (!data) return;
  const traces = [];
  if ($("showMask").checked && data.mesh.x.length) {
    traces.push({ type: "mesh3d", name: "mask", ...data.mesh, color: "#d62728",
                  opacity: parseFloat($("opacity").value), hoverinfo: "skip", showlegend: true });
  }
  if ($("showEdges").checked) traces.push(edgeTrace($("colorBy").value));
  if ($("showNodes").checked) traces.push(...nodeTraces());

  const [sx, sy, sz] = data.shape, m = Math.max(sx, sy, sz);
  const layout = {
    uirevision: current,
    margin: { l: 0, r: 0, t: 0, b: 0 },
    legend: { x: 0, y: 1 },
    scene: {
      xaxis: { title: "x", range: [0, sx - 1] },
      yaxis: { title: "y", range: [0, sy - 1] },
      zaxis: { title: "z", range: [0, sz - 1] },
      aspectmode: "manual",
      aspectratio: { x: sx / m, y: sy / m, z: sz / m },
    },
  };
  Plotly.react("plot", traces, layout, { responsive: true, displaylogo: false });
}

$("sample").onchange = e => loadSample(e.target.value);
["showMask", "showEdges", "showNodes", "colorBy"].forEach(id => $(id).onchange = render);
$("opacity").oninput = render;
loadSamples();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help="Sample folders directory, default is builder_output_dir from generation_config.yaml")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address, default is 127.0.0.1 (reach it through an SSH tunnel)")
    parser.add_argument("--port", type=int, default=8050, help="Port, default is 8050")
    args = parser.parse_args()

    data_dir = Path(args.dir if args.dir else load_config("generation_config.yaml")["builder_output_dir"])
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    print(f"Serving {len(list_samples(data_dir))} samples from {data_dir.resolve()}")
    print(f"Open http://localhost:{args.port} (from your machine: ssh -L {args.port}:localhost:{args.port} <user>@<host>)")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(data_dir))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
