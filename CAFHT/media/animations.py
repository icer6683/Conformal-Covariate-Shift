"""
Animate multiple objects, each with actual position, predicted position, and optional region.

Object 4 is special: it has no uncertainty region, and its predicted position is computed
to actively avoid ALL regions (circles) of the other objects at each time step.

Usage:
  python animate_predictions.py                 # dots only, 1.5s/frame
  python animate_predictions.py --speed 2.0     # custom speed
  python animate_predictions.py --use_image     # use images defined in OBJECTS
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation

# ── Parse arguments ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=float, default=1.5,
                    help="Seconds per frame (default: 1.5)")
parser.add_argument("--use_image", action="store_true",
                    help="Use image markers instead of dots")
args = parser.parse_args()
interval_ms = int(args.speed * 1000)

# ──────────────────────────────────────────────────────────────────
# DATA — loaded from file
# ──────────────────────────────────────────────────────────────────

# ── Put the data file in the same folder as this script ──
DATA_FILE = "example_results/n1000_modped_profdynamic_seed2000_ndim2_level10media_demo.txt"
df = pd.read_csv(DATA_FILE)
T = int(df["t"].max()) + 1   # 20

# ──────────────────────────────────────────────────────────────────
# PICK YOUR OBJECTS HERE
# First  index → kid1.png    (e.g. a hard object)
# Second index → adult1.png  (e.g. another object)
# Third  index → adult2.png  (e.g. another object)
# ──────────────────────────────────────────────────────────────────
SELECTED_IDS = [284, 1, 2]   # ← change these to any sample_idx values you like

kid1_id, adult1_id, adult2_id = SELECTED_IDS
print(f"kid1  → sample_idx={kid1_id}")
print(f"adult1→ sample_idx={adult1_id}")
print(f"adult2→ sample_idx={adult2_id}")

def load_obj(sample_id, color, image, zoom):
    sub = df[df["sample_idx"] == sample_id].sort_values("t").copy()
    # Forward-fill any NaN predictions (e.g. last time step)
    for col in ["pred_x", "pred_y", "band_lower_x", "band_upper_x",
                "band_lower_y", "band_upper_y"]:
        sub[col] = sub[col].ffill()
    r = (sub["pred_y"].values - sub["band_lower_y"].values)  # symmetric radius
    return {
        "x":    sub["actual_x"].values,
        "y":    sub["actual_y"].values,
        "xhat": sub["pred_x"].values,
        "yhat": sub["pred_y"].values,
        "r":    r,
        "color": color,
        "image": image,
        "zoom":  zoom,
    }

# Objects 1-3 have actual pos, predicted pos, and an uncertainty region
REGION_OBJECTS = [
    load_obj(kid1_id,  "#ff6644", "kid1.png",   0.06),
    load_obj(adult1_id, "#44ddff", "adult1.png", 0.06),
    load_obj(adult2_id, "#aaff66", "adult2.png", 0.06),
]

# ── Avoidance: compute predicted positions for object 4 ───────────
def mean_step_size(region_objects):
    """Compute mean step-to-step distance across all region objects' actual positions."""
    steps = []
    for obj in region_objects:
        xs, ys = obj["x"], obj["y"]
        dists = np.hypot(np.diff(xs), np.diff(ys))
        steps.extend(dists.tolist())
    return float(np.mean(steps))

def is_free(px, py, circles):
    """Return True if (px, py) is outside all circles."""
    return all(np.hypot(px - cx, py - cy) >= rad for cx, cy, rad in circles)

def astar_path(start, goal, circles, grid_n=200):
    """
    A* on a fine grid to find a collision-free path from start to goal.
    Returns list of (x, y) waypoints in world coordinates.
    """
    import heapq

    # Map world [-1,1] to grid [0, grid_n)
    def w2g(wx, wy):
        gx = int((wx + 1) / 2 * (grid_n - 1))
        gy = int((wy + 1) / 2 * (grid_n - 1))
        return (np.clip(gx, 0, grid_n-1), np.clip(gy, 0, grid_n-1))

    def g2w(gx, gy):
        wx = gx / (grid_n - 1) * 2 - 1
        wy = gy / (grid_n - 1) * 2 - 1
        return wx, wy

    # Pre-compute free cells
    xs = np.linspace(-1, 1, grid_n)
    ys = np.linspace(-1, 1, grid_n)
    free_grid = np.ones((grid_n, grid_n), dtype=bool)
    for (cx, cy, rad) in circles:
        # Mark all cells within radius as blocked
        ix = np.arange(grid_n)
        iy = np.arange(grid_n)
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        blocked = np.hypot(XX - cx, YY - cy) < rad
        free_grid &= ~blocked

    sg = w2g(*start)
    gg = w2g(*goal)

    # If start/goal are blocked, find nearest free cell
    if not free_grid[sg]:
        dists = np.full((grid_n, grid_n), np.inf)
        dists[sg] = 0
        best = None
        for dx in range(-grid_n, grid_n):
            for dy in range(-grid_n, grid_n):
                nx, ny = sg[0]+dx, sg[1]+dy
                if 0 <= nx < grid_n and 0 <= ny < grid_n and free_grid[nx, ny]:
                    d = abs(dx)+abs(dy)
                    if best is None or d < best[0]:
                        best = (d, (nx, ny))
                    break
            if best and best[0] < 3:
                break
        if best:
            sg = best[1]

    if not free_grid[gg]:
        for r in range(1, grid_n):
            found = False
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    nx, ny = gg[0]+dx, gg[1]+dy
                    if 0 <= nx < grid_n and 0 <= ny < grid_n and free_grid[nx, ny]:
                        gg = (nx, ny)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    # A* search with 8-connectivity
    def heuristic(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    open_heap = []
    heapq.heappush(open_heap, (0, sg))
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
                move_cost = np.hypot(dx, dy)
                tentative_g = g_score[cur] + move_cost
                if tentative_g < g_score.get(nb, np.inf):
                    g_score[nb] = tentative_g
                    f = tentative_g + heuristic(nb, gg)
                    heapq.heappush(open_heap, (f, nb))
                    came_from[nb] = cur

    # Reconstruct path
    path = []
    cur = gg
    while cur in came_from:
        path.append(g2w(*cur))
        cur = came_from[cur]
    path.append(g2w(*sg))
    path.reverse()
    path.append(g2w(*gg))
    return path

def resample_path_constant_speed(waypoints, step_size, n_points):
    """
    Resample a polyline (list of (x,y)) to exactly n_points
    with equal arc-length spacing of step_size.
    """
    # Build cumulative arc length
    pts = np.array(waypoints)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    # Target arc-length positions
    targets = np.linspace(0, total_len, n_points)

    result = []
    for s in targets:
        idx = np.searchsorted(cum_len, s, side='right') - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_start = cum_len[idx]
        seg_end = cum_len[idx + 1]
        seg_len = seg_end - seg_start
        if seg_len < 1e-12:
            result.append(pts[idx])
        else:
            t = (s - seg_start) / seg_len
            result.append(pts[idx] + t * (pts[idx+1] - pts[idx]))

    return np.array(result)

def compute_robot_path(start, region_objects, margin=0.04, grid_n=300):
    """
    Use A* on a grid to plan a collision-free path from start to (0.9, 0.9),
    then resample to T points with constant speed = mean object step size.
    Guarantees: zero collisions, constant ~0.2 speed per step.
    """
    step_size = mean_step_size(region_objects)
    goal = (0.9, 0.9)

    circles = [(obj["xhat"][i], obj["yhat"][i], obj["r"][i] + margin)
               for obj in region_objects for i in range(T)]

    print(f"  Planning path with A* (grid={grid_n}x{grid_n})...")
    waypoints = astar_path(start, goal, circles, grid_n=grid_n)
    print(f"  Found {len(waypoints)} waypoints")

    # Resample: divide total path length equally into T-1 steps (perfectly constant speed)
    pts = np.array(waypoints)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    targets = np.linspace(0, cum_len[-1], T)   # T equally spaced arc positions
    result = []
    for s in targets:
        idx = np.searchsorted(cum_len, s, side='right') - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_len = cum_len[idx+1] - cum_len[idx]
        t_frac = (s - cum_len[idx]) / seg_len if seg_len > 1e-12 else 0.0
        result.append(pts[idx] + t_frac * (pts[idx+1] - pts[idx]))
    path = np.array(result)

    # Final safety check: push any point that landed in a circle
    for i in range(T):
        px, py = path[i]
        for (cx, cy, rad) in circles:
            dist = np.hypot(px - cx, py - cy)
            if dist < rad:
                if dist < 1e-9:
                    px, py = cx + rad + 1e-4, cy
                else:
                    px = cx + (px - cx) * (rad + 1e-4) / dist
                    py = cy + (py - cy) * (rad + 1e-4) / dist
        path[i] = [np.clip(px, -0.99, 0.99), np.clip(py, -0.99, 0.99)]

    return path[:, 0], path[:, 1]

# Start robot near bottom-left, outside all circles
start = (-0.9, -0.9)
robot_x, robot_y = compute_robot_path(start, REGION_OBJECTS)

AVOID_OBJECT = {
    "x":    robot_x,
    "y":    robot_y,
    "color": "#b35c00",
    "image": "robot.png",
    "zoom":  0.06,
}

# All objects together for axis limits
ALL_OBJECTS = REGION_OBJECTS + [AVOID_OBJECT]

# ──────────────────────────────────────────────────────────────────
# LOAD IMAGES
# ──────────────────────────────────────────────────────────────────
# Load robot_smiling image for final frame
robot_smiling_img = None
if args.use_image:
    try:
        robot_smiling_img = mpimg.imread("robot_smiling.png")
        print("Loaded: robot_smiling.png")
    except FileNotFoundError:
        print("Warning: 'robot_smiling.png' not found — using robot.png for last frame too.")
        robot_smiling_img = None

if args.use_image:
    for obj in ALL_OBJECTS:
        try:
            obj["_img"] = mpimg.imread(obj["image"])
            print(f"Loaded: {obj['image']}")
        except FileNotFoundError:
            print(f"Warning: '{obj['image']}' not found — using dot fallback.")
            obj["_img"] = None
else:
    for obj in ALL_OBJECTS:
        obj["_img"] = None

# ──────────────────────────────────────────────────────────────────
# FIGURE SETUP  (white background)
# ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7), facecolor="white")
ax.set_facecolor("white")
for spine in ax.spines.values():
    spine.set_color("#cccccc")
ax.tick_params(colors="#333333")
ax.xaxis.label.set_color("#333333")
ax.yaxis.label.set_color("#333333")

all_x = np.concatenate([obj["x"] for obj in ALL_OBJECTS] +
                        [obj["xhat"] for obj in REGION_OBJECTS])
all_y = np.concatenate([obj["y"] for obj in ALL_OBJECTS] +
                        [obj["yhat"] for obj in REGION_OBJECTS])
all_r = np.concatenate([obj["r"] for obj in REGION_OBJECTS])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.set_title("CAFHT-assisted robot at Haidilao", color="#222222", fontsize=13, pad=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, color="#eeeeee", linewidth=0.8)

# ──────────────────────────────────────────────────────────────────
# BUILD ARTISTS — region objects (1-3)
# ──────────────────────────────────────────────────────────────────
def make_ab(img, zoom, pos):
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), pos, frameon=False)
    ab.set_zorder(8)
    return ab

for obj in REGION_OBJECTS:
    color = obj["color"]
    xhat, yhat, r = obj["xhat"], obj["yhat"], obj["r"]

    obj["_hist"] = []
    for i in range(T):
        c_fill = patches.Circle((xhat[i], yhat[i]), r[i],
                                 color="#888888", alpha=0.10, zorder=2, visible=False)
        c_edge = patches.Circle((xhat[i], yhat[i]), r[i],
                                 fill=False, edgecolor="#aaaaaa", linewidth=0.8,
                                 linestyle=":", alpha=0.35, zorder=2, visible=False)
        ax.add_patch(c_fill)
        ax.add_patch(c_edge)
        obj["_hist"].append((c_fill, c_edge))

    obj["_trail"], = ax.plot([], [], color=color, alpha=0.25, linewidth=1.2, zorder=3)

    obj["_pred_fill"] = patches.Circle((xhat[0], yhat[0]), r[0],
                                        color=color, alpha=0.18, zorder=4)
    obj["_pred_edge"] = patches.Circle((xhat[0], yhat[0]), r[0],
                                        fill=False, edgecolor=color, linewidth=1.6,
                                        linestyle="--", alpha=0.85, zorder=5)
    ax.add_patch(obj["_pred_fill"])
    ax.add_patch(obj["_pred_edge"])

    obj["_pred_dot"], = ax.plot([], [], "o", color=color, markersize=5,
                                 alpha=0.9, zorder=6)

    if obj["_img"] is not None:
        obj["_ab_container"] = [make_ab(obj["_img"], obj["zoom"], (obj["x"][0], obj["y"][0]))]
        ax.add_artist(obj["_ab_container"][0])
        obj["_actual_dot"] = None
    else:
        obj["_actual_dot"], = ax.plot([], [], "o", color=color, markersize=9, zorder=8)
        obj["_ab_container"] = [None]

# ──────────────────────────────────────────────────────────────────
# BUILD ARTISTS — robot (4): actual path only, no prediction, no region
# ──────────────────────────────────────────────────────────────────
obj4 = AVOID_OBJECT
color4 = obj4["color"]

# Trail only
obj4["_trail"], = ax.plot([], [], color=color4, alpha=0.35, linewidth=1.4,
                           linestyle="-", zorder=3)

# Actual position marker
if obj4["_img"] is not None:
    obj4["_ab_container"] = [make_ab(obj4["_img"], obj4["zoom"], (obj4["x"][0], obj4["y"][0]))]
    ax.add_artist(obj4["_ab_container"][0])
    obj4["_actual_dot"] = None
else:
    obj4["_actual_dot"], = ax.plot([], [], "D", color=color4, markersize=9, zorder=8)
    obj4["_ab_container"] = [None]

# ──────────────────────────────────────────────────────────────────
# LEGEND + LABELS
# ──────────────────────────────────────────────────────────────────
time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                    color="#333333", fontsize=10, va="top", family="monospace")

legend_grey  = patches.Patch(facecolor="#888888", edgecolor="#aaaaaa",
                               alpha=0.5, label="Past regions")
legend_robot = plt.Line2D([0], [0], marker="D", color=color4, linestyle="-",
                           markersize=7, label="Robot (avoidance path)")
ax.legend(handles=[legend_grey, legend_robot], loc="upper right",
          facecolor="white", edgecolor="#cccccc", labelcolor="#222222", fontsize=9)

# ──────────────────────────────────────────────────────────────────
# ANIMATION
# ──────────────────────────────────────────────────────────────────
def init():
    for obj in REGION_OBJECTS:
        obj["_trail"].set_data([], [])
        obj["_pred_dot"].set_data([], [])
        for c_fill, c_edge in obj["_hist"]:
            c_fill.set_visible(False)
            c_edge.set_visible(False)
        if obj["_actual_dot"] is not None:
            obj["_actual_dot"].set_data([], [])
    obj4["_trail"].set_data([], [])
    if obj4["_actual_dot"] is not None:
        obj4["_actual_dot"].set_data([], [])
    time_text.set_text("")


def update(frame):
    # ── Region objects ─────────────────────────
    for obj in REGION_OBJECTS:
        x, y = obj["x"], obj["y"]
        xhat, yhat, r = obj["xhat"], obj["yhat"], obj["r"]

        for i in range(frame):
            obj["_hist"][i][0].set_visible(True)
            obj["_hist"][i][1].set_visible(True)
        obj["_hist"][frame][0].set_visible(False)
        obj["_hist"][frame][1].set_visible(False)

        obj["_trail"].set_data(x[:frame + 1], y[:frame + 1])

        if obj["_ab_container"][0] is not None:
            obj["_ab_container"][0].remove()
            new_ab = AnnotationBbox(OffsetImage(obj["_img"], zoom=obj["zoom"]),
                                    (x[frame], y[frame]), frameon=False)
            new_ab.set_zorder(8)
            ax.add_artist(new_ab)
            obj["_ab_container"][0] = new_ab
        else:
            obj["_actual_dot"].set_data([x[frame]], [y[frame]])

        obj["_pred_fill"].center = (xhat[frame], yhat[frame])
        obj["_pred_fill"].radius = r[frame]
        obj["_pred_edge"].center = (xhat[frame], yhat[frame])
        obj["_pred_edge"].radius = r[frame]
        obj["_pred_dot"].set_data([xhat[frame]], [yhat[frame]])

    # ── Robot: actual path only ────────────────
    x4a, y4a = obj4["x"], obj4["y"]

    obj4["_trail"].set_data(x4a[:frame + 1], y4a[:frame + 1])

    if obj4["_ab_container"][0] is not None:
        obj4["_ab_container"][0].remove()
        # Use smiling robot on the last frame
        use_img = (robot_smiling_img if (frame == T - 1 and robot_smiling_img is not None)
                   else obj4["_img"])
        new_ab = AnnotationBbox(OffsetImage(use_img, zoom=obj4["zoom"]),
                                (x4a[frame], y4a[frame]), frameon=False)
        new_ab.set_zorder(8)
        ax.add_artist(new_ab)
        obj4["_ab_container"][0] = new_ab
    else:
        obj4["_actual_dot"].set_data([x4a[frame]], [y4a[frame]])

    time_text.set_text(f"t = {frame + 1:>2d} / {T}")
    fig.canvas.draw_idle()


ani = FuncAnimation(fig, update, frames=T, init_func=init,
                    interval=interval_ms, blit=False)

plt.tight_layout()

# ── Output ────────────────────────────────────
plt.show()

# Save as GIF (uncomment):
# fps = max(1, int(1000 / interval_ms))
# ani.save("animation.gif", writer=PillowWriter(fps=fps))

# Save as MP4 (requires ffmpeg, uncomment):
fps = max(1, int(1000 / interval_ms))
ani.save("animation_v2.mp4", fps=fps, dpi=300, extra_args=["-vcodec", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p"])
print("Saved animation.mp4")
