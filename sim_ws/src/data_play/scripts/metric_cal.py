#!/usr/bin/env python3

import os
import numpy as np
import glob
import re
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from shapely.affinity import rotate, translate


class obstacle:
    def __init__(self, vertices):
        self.polygon = Polygon(vertices)
        self.robot_radius = 0.35

    def collsion(self, point):
        p = Point(point)
        return 1 if p.distance(self.polygon) < self.robot_radius else 0


class agent:
    def __init__(self, id, data, track_dict):
        self.robot_radius = 0.5
        self.human_radius = 0.5
        self.id = id
        if id < 0:
            self.type = 0
            return
        self.data = data
        if track_dict[id] in ("Car", "Bus"):
            self.type = 2
        elif track_dict[id] in ("Biker", "Cart"):
            self.type = 1
        else:
            self.type = 0

    def _create_footprint(self, px, py, yaw):
        if self.type == 0:
            return Point(px, py).buffer(self.human_radius)
        elif self.type == 1:
            rect = Polygon([(-0.9, -0.5), (0.9, -0.5), (0.9, 0.5), (-0.9, 0.5)])
        elif self.type == 2:
            rect = Polygon([(-2.25, -0.95), (2.25, -0.95), (2.25, 0.95), (-2.25, 0.95)])
        rotated_rect = rotate(rect, np.degrees(yaw), origin=(0, 0), use_radians=False)
        return translate(rotated_rect, xoff=px, yoff=py)

    def cal_dist(self, px, py, yaw, target_point):
        return Point(target_point).distance(self._create_footprint(px, py, yaw))

    def cal_index_collision(self, index, target_point):
        px, py, yaw = self.data[index, 1], self.data[index, 2], self.data[index, 3]
        return 1 if self.cal_dist(px, py, yaw, target_point) <= self.robot_radius else 0


def quaternion_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def read_actor_data(trial_folder):
    actor_data = {}
    for file in glob.glob(os.path.join(trial_folder, "actor_*.txt")):
        actor_id = int(re.search(r"actor_(\d+).txt", file).group(1))
        data = np.loadtxt(file)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        time, px, py, pz, qx, qy, qz, qw = data.T
        yaw_angles = np.array([quaternion_to_yaw(qx[i], qy[i], qz[i], qw[i]) for i in range(len(qx))])
        actor_data[actor_id] = np.column_stack((time, px, py, yaw_angles))
    return actor_data


def read_status_data(trial_folder):
    status_file = os.path.join(trial_folder, "status.txt")
    if not os.path.exists(status_file):
        raise FileNotFoundError(f"Status file not found: {status_file}")
    with open(status_file, "r") as f:
        lines = f.readlines()
    status_data = []
    for line in lines[:-1]:
        parts = line.strip().split()
        if len(parts) == 3:
            status_data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    status_data = np.array(status_data) if status_data else np.empty((0, 3))
    last_line = lines[-1].strip()
    finish_time = None
    if last_line.startswith("success:"):
        finish_time = float(last_line.split(":")[1].strip())
    return status_data, last_line, finish_time


def process_scene(scene_name):
    base_path = f"/root/test_data/{scene_name}/"
    trial_folders = glob.glob(os.path.join(base_path, "trial_*"))
    results = {}
    for trial_folder in trial_folders:
        trial_id = os.path.basename(trial_folder).split("_")[-1]
        if int(trial_id) < 0:
            continue
        actor_data = read_actor_data(trial_folder)
        status_data, last_status_message, finish_time = read_status_data(trial_folder)
        results[trial_id] = {
            "actor_data": actor_data,
            "status_data": status_data,
            "last_status_message": last_status_message,
            "finish_time": finish_time,
        }
    return results


# ── config ────────────────────────────────────────────────────────────────────
scene_name = "crossing_0"

scene_results = process_scene(scene_name)

track_dict = {}
with open(f"/root/sim_ws/src/data_play/temp/{scene_name}/pair/data_id_label_pairs.txt", "r") as file:
    next(file)
    for line in file:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            track_id, label = parts
            track_dict[int(track_id)] = label

config_path = '/root/sim_ws/src/data_play/dataset/scene_config_30/' + scene_name + '.json'
with open(config_path, 'r') as f:
    scene_config = json.load(f)

obs_list = []
for obs in scene_config["obstacles"]:
    obs_list.append(obstacle(np.array(obs)))

robot_pos_goal = np.array(scene_config["robot_start_end"])
start_pos = robot_pos_goal[0:2]
goal_pos  = robot_pos_goal[2:4]

# ── experiment log ────────────────────────────────────────────────────────────
file_path = f"/root/test_data/{scene_name}/experiment_log.csv"
if os.path.exists(file_path):
    os.remove(file_path)
with open(file_path, "w") as f:
    f.write("Trial_ID, Finish, Total_collision, traveling_dis, Total_Time\n")

# ── find most recent trial ────────────────────────────────────────────────────
sorted_trials = sorted(scene_results.items(), key=lambda x: int(x[0]))
latest_trial_id, latest_data = sorted_trials[-1]
print(f"Plotting most recent trial: {latest_trial_id}")

# ── process ALL trials for the CSV, but only plot the latest ─────────────────
for trial_id, data in sorted_trials:
    finish_flag = 1 if data["finish_time"] is not None else 0
    status = data["status_data"]

    agent_list = [agent(aid, arr, track_dict) for aid, arr in data["actor_data"].items()]

    human_collision = 0
    obs_collision   = 0
    traveling_dis   = 0
    previous_position = None

    for index, row in enumerate(status):
        robot_position = row[1:4]
        if previous_position is None:
            previous_position = robot_position
        traveling_dis += np.linalg.norm(robot_position - previous_position)
        previous_position = robot_position
        for ped in agent_list:
            if ped.cal_index_collision(index, robot_position) == 1:
                human_collision += 1
        for obs in obs_list:
            if obs.collsion(robot_position):
                obs_collision += 1

    with open(file_path, "a") as f:
        f.write(f"{trial_id},{finish_flag},{human_collision+obs_collision},{traveling_dis},{data['finish_time']}\n")

# ── plot latest trial only ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.set_xlabel("X")
ax.set_ylabel("Y")

finish_flag = 1 if latest_data["finish_time"] is not None else 0
outcome_str = f"success ({latest_data['finish_time']:.1f}s)" if finish_flag else "timeout"
ax.set_title(f"Trial {latest_trial_id} — {scene_name} — {outcome_str}")

# Draw obstacles
for obs in obs_list:
    x, y = obs.polygon.exterior.xy
    ax.fill(x, y, color='black', alpha=0.85, zorder=2)

status = latest_data["status_data"]

if len(status) > 1:
    xs = status[:, 1]
    ys = status[:, 2]

    # ── path line ────────────────────────────────────────────────────────────
    ax.plot(xs, ys, color='royalblue', linewidth=2.0, zorder=3, label='Robot path')

    # ── orientation arrows ────────────────────────────────────────────────────
    # Estimate yaw at each point from consecutive position differences.
    # We skip the last point since it has no "next" point to diff against.
    # Arrow spacing: every N frames so the plot stays readable.
    arrow_every = max(1, len(xs) // 30)  # ~30 arrows regardless of path length
    arrow_len = 0.4  # metres

    dx = np.diff(xs)
    dy = np.diff(ys)
    norms = np.sqrt(dx**2 + dy**2)
    # Avoid division by zero for stationary frames
    norms[norms == 0] = 1e-9
    dx_norm = dx / norms
    dy_norm = dy / norms

    for i in range(0, len(dx), arrow_every):
        ax.annotate(
            "",
            xy=(xs[i] + dx_norm[i] * arrow_len, ys[i] + dy_norm[i] * arrow_len),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="-|>", color='darkorange', lw=1.5),
            zorder=4,
        )

    # ── start / end markers ───────────────────────────────────────────────────
    ax.scatter(xs[0],  ys[0],  color='green', marker='o', s=100, zorder=5, label='Path start')
    end_marker = '*' if finish_flag else 'x'
    ax.scatter(xs[-1], ys[-1], color='red',   marker=end_marker, s=150, zorder=5,
               label='Path end (success)' if finish_flag else 'Path end (timeout)')

# Scene start / goal
ax.scatter(*start_pos, color='green', marker='^', s=220, zorder=6, label='Scene start')
ax.scatter(*goal_pos,  color='red',   marker='*', s=220, zorder=6, label='Goal')

ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

out_path = f"/root/test_data/{scene_name}/robot_paths.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved plot to: {out_path}")