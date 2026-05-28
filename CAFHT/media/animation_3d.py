"""
3D animation: actual vs predicted positions with uncertainty regions.
- X, Y from real data; Z simulated as smooth sinusoids per object.
- Prediction regions are horizontal disks (circles in XY at object's Z).
- Robot navigates from bottom-left-bottom to upper-right-top avoiding all regions.

Usage:
  python animate_predictions.py                 # dots only, 1.5s/frame
  python animate_predictions.py --speed 2.0     # custom speed
  python animate_predictions.py --use_image     # use images as markers
"""

import argparse
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D, proj3d

# ── Parse arguments ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=float, default=1.5,
                    help="Seconds per frame (default: 1.5)")
# Images disabled — dots only
args = parser.parse_args()
interval_ms = int(args.speed * 1000)

# ──────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────
DATA_FILE = "example_results/n1000_modped_profdynamic_seed2000_ndim2_level10media_demo.txt"
df = pd.read_csv(DATA_FILE)
T = int(df["t"].max()) + 1   # 20

# ──────────────────────────────────────────────────────────────────
# PICK YOUR OBJECTS HERE
# First  index → kid1.png
# Second index → adult1.png
# Third  index → adult2.png
# ──────────────────────────────────────────────────────────────────
SELECTED_IDS = [284, 1, 2]
kid1_id, adult1_id, adult2_id = SELECTED_IDS
print(f"kid1  → sample_idx={kid1_id}")
print(f"adult1→ sample_idx={adult1_id}")
print(f"adult2→ sample_idx={adult2_id}")

def load_obj(sample_id, color, image, zoom, z_amp, z_freq, z_phase):
    sub = df[df["sample_idx"] == sample_id].sort_values("t").copy()
    for col in ["pred_x", "pred_y", "band_lower_x", "band_upper_x",
                "band_lower_y", "band_upper_y"]:
        sub[col] = sub[col].ffill()
    r = sub["pred_y"].values - sub["band_lower_y"].values

    # Simulate Z as a smooth sinusoid
    t_norm = np.linspace(0, 2 * np.pi, T)
    z    = z_amp * np.sin(z_freq * t_norm + z_phase)
    zhat = z_amp * np.sin(z_freq * (t_norm + 0.2) + z_phase)

    return {
        "x":    sub["actual_x"].values,
        "y":    sub["actual_y"].values,
        "z":    z,
        "xhat": sub["pred_x"].values,
        "yhat": sub["pred_y"].values,
        "zhat": zhat,
        "r":    r,
        "color": color,
        "image": image,
        "zoom":  zoom,
    }

REGION_OBJECTS = [
    load_obj(kid1_id,   "#ff6644", "kid1.png",   0.06, z_amp=0.35, z_freq=1.5, z_phase=0.0),
    load_obj(adult1_id, "#44ddff", "adult1.png", 0.06, z_amp=0.28, z_freq=1.2, z_phase=1.1),
    load_obj(adult2_id, "#aaff66", "adult2.png", 0.06, z_amp=0.22, z_freq=1.8, z_phase=2.3),
]

# ──────────────────────────────────────────────────────────────────
# ROBOT PATH — A* in XY, Z interpolated bottom→top
# ──────────────────────────────────────────────────────────────────
def mean_step_size(region_objects):
    steps = []
    for obj in region_objects:
        steps.extend(np.hypot(np.diff(obj["x"]), np.diff(obj["y"])).tolist())
    return float(np.mean(steps))

def astar_path(start, goal, circles, grid_n=300):
    def w2g(wx, wy):
        gx = int((wx + 1) / 2 * (grid_n - 1))
        gy = int((wy + 1) / 2 * (grid_n - 1))
        return (np.clip(gx, 0, grid_n-1), np.clip(gy, 0, grid_n-1))

    def g2w(gx, gy):
        return gx / (grid_n - 1) * 2 - 1, gy / (grid_n - 1) * 2 - 1

    xs = np.linspace(-1, 1, grid_n)
    ys = np.linspace(-1, 1, grid_n)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    free_grid = np.ones((grid_n, grid_n), dtype=bool)
    for (cx, cy, rad) in circles:
        free_grid &= (np.hypot(XX - cx, YY - cy) >= rad)

    sg, gg = w2g(*start), w2g(*goal)

    open_heap = [(0, sg)]
    came_from = {}
    g_score = {sg: 0}

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == gg:
            break
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nb = (cur[0]+dx, cur[1]+dy)
                if not (0 <= nb[0] < grid_n and 0 <= nb[1] < grid_n):
                    continue
                if not free_grid[nb]:
                    continue
                tg = g_score[cur] + np.hypot(dx, dy)
                if tg < g_score.get(nb, np.inf):
                    g_score[nb] = tg
                    heapq.heappush(open_heap, (tg + np.hypot(nb[0]-gg[0], nb[1]-gg[1]), nb))
                    came_from[nb] = cur

    path = []
    cur = gg
    while cur in came_from:
        path.append(g2w(*cur))
        cur = came_from[cur]
    path.append(g2w(*sg))
    path.reverse()
    path.append(g2w(*gg))
    return path

def resample_equal(waypoints, n):
    pts = np.array(waypoints)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    targets = np.linspace(0, cum_len[-1], n)
    result = []
    for s in targets:
        idx = np.clip(np.searchsorted(cum_len, s, side='right') - 1, 0, len(pts)-2)
        sl = cum_len[idx+1] - cum_len[idx]
        tf = (s - cum_len[idx]) / sl if sl > 1e-12 else 0.0
        result.append(pts[idx] + tf * (pts[idx+1] - pts[idx]))
    return np.array(result)

margin = 0.04
circles_xy = [(obj["xhat"][i], obj["yhat"][i], obj["r"][i] + margin)
              for obj in REGION_OBJECTS for i in range(T)]

print("  Running A* path planning...")
waypoints = astar_path((-0.9, -0.9), (0.9, 0.9), circles_xy)
robot_xy  = resample_equal(waypoints, T)

# Push any resampled point clear of circles
for i in range(T):
    px, py = robot_xy[i]
    for (cx, cy, rad) in circles_xy:
        dist = np.hypot(px-cx, py-cy)
        if dist < rad:
            px = cx + (px-cx) * (rad+1e-4) / dist if dist > 1e-9 else cx+rad+1e-4
            py = cy + (py-cy) * (rad+1e-4) / dist if dist > 1e-9 else cy
    robot_xy[i] = [np.clip(px, -0.99, 0.99), np.clip(py, -0.99, 0.99)]

robot_x = robot_xy[:, 0]
robot_y = robot_xy[:, 1]
robot_z = np.linspace(-0.7, 0.7, T)   # Z: bottom→top

# Verify
steps = np.hypot(np.diff(robot_x), np.diff(robot_y))
collisions = sum(np.hypot(robot_x[t]-obj["xhat"][t], robot_y[t]-obj["yhat"][t]) < obj["r"][t]
                 for t in range(T) for obj in REGION_OBJECTS)
print(f"  Robot step mean: {steps.mean():.4f}, std: {steps.std():.4f}")
print(f"  Collisions: {collisions}")

AVOID_OBJECT = {
    "x": robot_x, "y": robot_y, "z": robot_z,
    "color": "#ffdd44", "image": "robot.png", "zoom": 0.06,
}

ALL_OBJECTS = REGION_OBJECTS + [AVOID_OBJECT]

# ──────────────────────────────────────────────────────────────────
# LOAD IMAGES
# ──────────────────────────────────────────────────────────────────
robot_smiling_img = None
for obj in ALL_OBJECTS:
    obj["_img"] = None

# ──────────────────────────────────────────────────────────────────
# FIGURE — 3D
# ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 8), facecolor="white")
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor("white")
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#dddddd")
ax.yaxis.pane.set_edgecolor("#dddddd")
ax.zaxis.pane.set_edgecolor("#dddddd")
ax.tick_params(colors="#333333", labelsize=7)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_xlabel("x", color="#333333", labelpad=4)
ax.set_ylabel("y", color="#333333", labelpad=4)
ax.set_zlabel("z", color="#333333", labelpad=4)
ax.set_title("CAFHT-assisted robot (3D)", color="#111111", fontsize=13, pad=10)
ax.view_init(elev=22, azim=-55)
ax.xaxis._axinfo["grid"]["color"] = "#dddddd"
ax.yaxis._axinfo["grid"]["color"] = "#dddddd"
ax.zaxis._axinfo["grid"]["color"] = "#dddddd"

# ── Helpers ───────────────────────────────────────────────────────
def make_sphere_surface(cx, cy, cz, r, n=24):
    """Return X, Y, Z arrays for a sphere surface (for plot_surface)."""
    u = np.linspace(0, 2*np.pi, n)
    v = np.linspace(0, np.pi, n)
    X = cx + r * np.outer(np.cos(u), np.sin(v))
    Y = cy + r * np.outer(np.sin(u), np.sin(v))
    Z = cz + r * np.outer(np.ones(n), np.cos(v))
    return X, Y, Z

def make_ab(img, zoom, xy2d):
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), xy2d, frameon=False,
                        xycoords='data', boxcoords='data')
    ab.set_zorder(10)
    return ab

def proj2d(x3, y3, z3):
    """Project 3D coords to 2D axes data coords using current view."""
    x2, y2, _ = proj3d.proj_transform(x3, y3, z3, ax.get_proj())
    return x2, y2

# ──────────────────────────────────────────────────────────────────
# BUILD ARTISTS — region objects
# ──────────────────────────────────────────────────────────────────
for obj in REGION_OBJECTS:
    color = obj["color"]
    xhat, yhat, zhat, r = obj["xhat"], obj["yhat"], obj["zhat"], obj["r"]

    # History spheres (pre-created, toggled visible)
    obj["_hist"] = []
    for i in range(T):
        SX, SY, SZ = make_sphere_surface(xhat[i], yhat[i], zhat[i], r[i], n=16)
        sphere = ax.plot_surface(SX, SY, SZ, color="#888888", alpha=0.07,
                                 linewidth=0, antialiased=False)
        sphere.set_visible(False)
        obj["_hist"].append((sphere,))

    # Trail
    obj["_trail"], = ax.plot3D([], [], [], color=color, alpha=0.85,
                               linewidth=2.0, zorder=3)

    # Current prediction: container (remove+recreate each frame)
    obj["_pred_container"] = [None]

    obj["_actual_dot_container"] = [None]

# ──────────────────────────────────────────────────────────────────
# BUILD ARTISTS — robot
# ──────────────────────────────────────────────────────────────────
obj4    = AVOID_OBJECT
color4  = obj4["color"]

obj4["_trail"], = ax.plot3D([], [], [], color=color4, alpha=0.9,
                             linewidth=2, zorder=10)

obj4["_actual_dot_container"] = [None]
obj4["_ab_container"] = [None]

# ──────────────────────────────────────────────────────────────────
# LABELS
# ──────────────────────────────────────────────────────────────────
time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes,
                      color="#333333", fontsize=10, va="top", family="monospace")
ax.text2D(0.02, 0.91, f"speed: {args.speed}s / frame", transform=ax.transAxes,
          color="#555555", fontsize=8, va="top", family="monospace")

# ──────────────────────────────────────────────────────────────────
# ANIMATION
# ──────────────────────────────────────────────────────────────────
def remove_if(container):
    if container[0] is not None:
        try:
            container[0].remove()
        except Exception:
            pass
        container[0] = None

def add_image_marker(img, zoom, x3, y3, z3):
    x2d, y2d = proj2d(x3, y3, z3)
    ab = AnnotationBbox(
        OffsetImage(img, zoom=zoom),
        (x2d, y2d), frameon=False, xycoords='data', boxcoords='data'
    )
    ab.set_zorder(10)
    return ax.add_artist(ab)

def plot3d_dot(x, y, z, color, marker="o", size=60):
    return ax.scatter([x], [y], [z], color=color, s=size,
                      depthshade=True, zorder=8)

def init():
    for obj in REGION_OBJECTS:
        obj["_trail"].set_data_3d([], [], [])
        for (sphere,) in obj["_hist"]:
            sphere.set_visible(False)
        remove_if(obj["_actual_dot_container"])
        remove_if(obj["_pred_container"])
    obj4["_trail"].set_data_3d([], [], [])
    remove_if(obj4["_actual_dot_container"])
    time_text.set_text("")


def update(frame):
    for obj in REGION_OBJECTS:
        x, y, z       = obj["x"], obj["y"], obj["z"]
        xhat, yhat, zhat, r = obj["xhat"], obj["yhat"], obj["zhat"], obj["r"]
        color         = obj["color"]

        # History: reveal past spheres
        for i in range(frame):
            obj["_hist"][i][0].set_visible(True)
        obj["_hist"][frame][0].set_visible(False)

        # Trail
        obj["_trail"].set_data_3d(x[:frame+1], y[:frame+1], z[:frame+1])

        # Current prediction sphere — remove old, add new
        if obj["_pred_container"][0] is not None:
            obj["_pred_container"][0].remove()
        SX, SY, SZ = make_sphere_surface(xhat[frame], yhat[frame],
                                          zhat[frame], r[frame], n=28)
        sphere = ax.plot_surface(SX, SY, SZ, color=color, alpha=0.22,
                                 linewidth=0, antialiased=True)
        obj["_pred_container"][0] = sphere

        # Actual position dot
        remove_if(obj["_actual_dot_container"])
        obj["_actual_dot_container"][0] = plot3d_dot(
            x[frame], y[frame], z[frame], color, size=80)

    # Robot
    x4, y4, z4 = obj4["x"], obj4["y"], obj4["z"]
    obj4["_trail"].set_data_3d(x4[:frame+1], y4[:frame+1], z4[:frame+1])

    remove_if(obj4["_actual_dot_container"])
    obj4["_actual_dot_container"][0] = plot3d_dot(
        x4[frame], y4[frame], z4[frame], color4, marker="D", size=90)

    time_text.set_text(f"t = {frame+1:>2d} / {T}")
    fig.canvas.draw_idle()


ani = FuncAnimation(fig, update, frames=T, init_func=init,
                    interval=interval_ms, blit=False)

plt.tight_layout()

# Uncomment to show interactive window instead:
plt.show()

# Uncomment to save as GIF:
# from matplotlib.animation import PillowWriter
# ani.save("animation.gif", writer=PillowWriter(fps=fps))

# ── Save as MP4 ───────────────────────────────
fps = max(1, int(1000 / interval_ms))
ani.save("animation_3d.mp4", fps=fps, dpi=300,
         extra_args=["-vcodec", "libx264", "-crf", "1",
                     "-preset", "veryslow", "-pix_fmt", "yuv420p"])
print("Saved animation_3d.mp4")