"""
Animate multiple objects, each with actual position, predicted position, and optional region.

Object 4 is special: it has no uncertainty region, and its predicted position is computed
to actively avoid ALL regions (rectangles) of the other objects at each time step.

Usage:
  python animate_predictions_rect.py                 # dots only, 1.5s/frame
  python animate_predictions_rect.py --speed 2.0     # custom speed
  python animate_predictions_rect.py --use_image     # use images defined in OBJECTS
"""

import argparse
import heapq
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
DATA_FILE = "example_results/n1000_modped_profdynamic_seed2000_ndim2_level10media_demo.txt"
df = pd.read_csv(DATA_FILE)
T = int(df["t"].max()) + 1   # 20

SELECTED_IDS = [284, 1, 2]
kid1_id, adult1_id, adult2_id = SELECTED_IDS
print(f"kid1  → sample_idx={kid1_id}")
print(f"adult1→ sample_idx={adult1_id}")
print(f"adult2→ sample_idx={adult2_id}")

def load_obj(sample_id, color, image, zoom):
    sub = df[df["sample_idx"] == sample_id].sort_values("t").copy()
    for col in ["pred_x", "pred_y", "band_lower_x", "band_upper_x",
                "band_lower_y", "band_upper_y"]:
        sub[col] = sub[col].ffill()
    return {
        "x":    sub["actual_x"].values,
        "y":    sub["actual_y"].values,
        "xhat": sub["pred_x"].values,
        "yhat": sub["pred_y"].values,
        "xl":   sub["band_lower_x"].values,
        "xu":   sub["band_upper_x"].values,
        "yl":   sub["band_lower_y"].values,
        "yu":   sub["band_upper_y"].values,
        "color": color,
        "image": image,
        "zoom":  zoom,
    }

REGION_OBJECTS = [
    load_obj(kid1_id,   "#ff6644", "kid1.png",   0.06),
    load_obj(adult1_id, "#44ddff", "adult1.png", 0.06),
    load_obj(adult2_id, "#aaff66", "adult2.png", 0.06),
]

# ── Robot path — A* avoiding all rectangles ───────────────────────
def mean_step_size(region_objects):
    steps = []
    for obj in region_objects:
        steps.extend(np.hypot(np.diff(obj["x"]), np.diff(obj["y"])).tolist())
    return float(np.mean(steps))

def astar_path(start, goal, rects, grid_n=300):
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
    for (xl, xu, yl, yu) in rects:
        free_grid &= ~((XX >= xl) & (XX <= xu) & (YY >= yl) & (YY <= yu))

    sg, gg = w2g(*start), w2g(*goal)
    for target, label in [(sg, 's'), (gg, 'g')]:
        if not free_grid[target]:
            for r in range(1, grid_n):
                found = False
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        nx, ny = target[0]+dx, target[1]+dy
                        if 0 <= nx < grid_n and 0 <= ny < grid_n and free_grid[nx, ny]:
                            if label == 's': sg = (nx, ny)
                            else: gg = (nx, ny)
                            found = True; break
                    if found: break
                if found: break

    open_heap = [(0, sg)]
    came_from = {}
    g_score = {sg: 0}
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == gg: break
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                nb = (cur[0]+dx, cur[1]+dy)
                if not (0 <= nb[0] < grid_n and 0 <= nb[1] < grid_n): continue
                if not free_grid[nb]: continue
                tg = g_score[cur] + np.hypot(dx, dy)
                if tg < g_score.get(nb, np.inf):
                    g_score[nb] = tg
                    heapq.heappush(open_heap, (tg + np.hypot(nb[0]-gg[0], nb[1]-gg[1]), nb))
                    came_from[nb] = cur

    path = []
    cur = gg
    while cur in came_from:
        path.append(g2w(*cur)); cur = came_from[cur]
    path.append(g2w(*sg)); path.reverse(); path.append(g2w(*gg))
    return path

def compute_robot_path(start, region_objects, margin=0.04, grid_n=300):
    goal = (0.9, 0.9)
    rects = [(obj["xl"][i] - margin, obj["xu"][i] + margin,
              obj["yl"][i] - margin, obj["yu"][i] + margin)
             for obj in region_objects for i in range(T)]

    print(f"  Planning path with A* (grid={grid_n}x{grid_n})...")
    waypoints = astar_path(start, goal, rects, grid_n=grid_n)
    print(f"  Found {len(waypoints)} waypoints")

    pts = np.array(waypoints)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    targets = np.linspace(0, cum_len[-1], T)
    result = []
    for s in targets:
        idx = np.clip(np.searchsorted(cum_len, s, side='right') - 1, 0, len(pts)-2)
        sl = cum_len[idx+1] - cum_len[idx]
        tf = (s - cum_len[idx]) / sl if sl > 1e-12 else 0.0
        result.append(pts[idx] + tf * (pts[idx+1] - pts[idx]))
    path = np.array(result)

    # Safety nudge out of any rectangle
    for i in range(T):
        px, py = path[i]
        for (xl, xu, yl, yu) in rects:
            if xl <= px <= xu and yl <= py <= yu:
                dists = [px - xl, xu - px, py - yl, yu - py]
                mi = int(np.argmin(dists))
                if mi == 0:   px = xl - 1e-4
                elif mi == 1: px = xu + 1e-4
                elif mi == 2: py = yl - 1e-4
                else:         py = yu + 1e-4
        path[i] = [np.clip(px, -0.99, 0.99), np.clip(py, -0.99, 0.99)]

    return path[:, 0], path[:, 1]

start = (-0.9, -0.9)
robot_x, robot_y = compute_robot_path(start, REGION_OBJECTS)

AVOID_OBJECT = {
    "x":    robot_x,
    "y":    robot_y,
    "color": "#b35c00",
    "image": "robot.png",
    "zoom":  0.06,
}
ALL_OBJECTS = REGION_OBJECTS + [AVOID_OBJECT]

# ──────────────────────────────────────────────────────────────────
# LOAD IMAGES
# ──────────────────────────────────────────────────────────────────
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
    xl, xu, yl, yu = obj["xl"], obj["xu"], obj["yl"], obj["yu"]

    # History rectangles
    obj["_hist"] = []
    for i in range(T):
        w, h = xu[i] - xl[i], yu[i] - yl[i]
        r_fill = patches.Rectangle((xl[i], yl[i]), w, h,
                                    facecolor="#888888", alpha=0.10,
                                    zorder=2, visible=False)
        r_edge = patches.Rectangle((xl[i], yl[i]), w, h,
                                    fill=False, edgecolor="#aaaaaa",
                                    linewidth=0.8, linestyle=":",
                                    alpha=0.35, zorder=2, visible=False)
        ax.add_patch(r_fill)
        ax.add_patch(r_edge)
        obj["_hist"].append((r_fill, r_edge))

    obj["_trail"], = ax.plot([], [], color=color, alpha=0.25, linewidth=1.2, zorder=3)

    # Live prediction rectangle
    w0, h0 = xu[0] - xl[0], yu[0] - yl[0]
    obj["_pred_fill"] = patches.Rectangle((xl[0], yl[0]), w0, h0,
                                           facecolor=color, alpha=0.18, zorder=4)
    obj["_pred_edge"] = patches.Rectangle((xl[0], yl[0]), w0, h0,
                                           fill=False, edgecolor=color,
                                           linewidth=1.6, linestyle="--",
                                           alpha=0.85, zorder=5)
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
# BUILD ARTISTS — robot (4)
# ──────────────────────────────────────────────────────────────────
obj4 = AVOID_OBJECT
color4 = obj4["color"]

obj4["_trail"], = ax.plot([], [], color=color4, alpha=0.35, linewidth=1.4,
                           linestyle="-", zorder=3)

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
        for r_fill, r_edge in obj["_hist"]:
            r_fill.set_visible(False)
            r_edge.set_visible(False)
        if obj["_actual_dot"] is not None:
            obj["_actual_dot"].set_data([], [])
    obj4["_trail"].set_data([], [])
    if obj4["_actual_dot"] is not None:
        obj4["_actual_dot"].set_data([], [])
    time_text.set_text("")


def update(frame):
    for obj in REGION_OBJECTS:
        x, y = obj["x"], obj["y"]
        xl, xu, yl, yu = obj["xl"], obj["xu"], obj["yl"], obj["yu"]

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

        # Update live rectangle
        w = xu[frame] - xl[frame]
        h = yu[frame] - yl[frame]
        obj["_pred_fill"].set_xy((xl[frame], yl[frame]))
        obj["_pred_fill"].set_width(w)
        obj["_pred_fill"].set_height(h)
        obj["_pred_edge"].set_xy((xl[frame], yl[frame]))
        obj["_pred_edge"].set_width(w)
        obj["_pred_edge"].set_height(h)

        obj["_pred_dot"].set_data([obj["xhat"][frame]], [obj["yhat"][frame]])

    # ── Robot ──────────────────────────────────
    x4a, y4a = obj4["x"], obj4["y"]
    obj4["_trail"].set_data(x4a[:frame + 1], y4a[:frame + 1])

    if obj4["_ab_container"][0] is not None:
        obj4["_ab_container"][0].remove()
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
ani.save("animation_v2_rect.mp4", fps=fps, dpi=300, extra_args=["-vcodec", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p"])
print("Saved animation_v2_rect.mp4")