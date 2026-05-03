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
        if track_dict.get(id) in ("Car", "Bus"):
            self.type = 2
        elif track_dict.get(id) in ("Biker", "Cart"):
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
        else:
            rect = Point(0, 0).buffer(self.human_radius)
            
        rotated_rect = rotate(rect, np.degrees(yaw), origin=(0, 0), use_radians=False)
        return translate(rotated_rect, xoff=px, yoff=py)

    def cal_dist(self, px, py, yaw, target_point):
        return Point(target_point).distance(self._create_footprint(px, py, yaw))

    def cal_index_collision(self, index, target_point):
        if index >= len(self.data):
            return 0
        px, py, yaw = self.data[index, 1], self.data[index, 2], self.data[index, 3]
        return 1 if self.cal_dist(px, py, yaw, target_point) <= self.robot_radius else 0


def quaternion_to_yaw(qx, qy, qz, qw):
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def read_actor_data(trial_folder):
    actor_data = {}
    for file in glob.glob(os.path.join(trial_folder, "actor_*.txt")):
        try:
            match = re.search(r"actor_(\d+).txt", file)
            if not match: continue
            actor_id = int(match.group(1))
            data = np.loadtxt(file)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            time, px, py, pz, qx, qy, qz, qw = data.T
            yaw_angles = np.array([quaternion_to_yaw(qx[i], qy[i], qz[i], qw[i]) for i in range(len(qx))])
            actor_data[actor_id] = np.column_stack((time, px, py, yaw_angles))
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")
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
        if len(parts) == 4:
            status_data.append([float(p) for p in parts])
        elif len(parts) == 3:
            status_data.append([float(p) for p in parts] + [np.nan])

    status_data = np.array(status_data, dtype=float) if status_data else np.empty((0, 4))
    last_line = lines[-1].strip()
    finish_time = float(last_line.split(":")[1].strip()) if "success:" in last_line else None
    return status_data, last_line, finish_time


def process_scene(scene_name):
    base_path = f"/root/test_data/{scene_name}/"
    trial_folders = glob.glob(os.path.join(base_path, "trial_*"))
    results = {}
    for trial_folder in trial_folders:
        try:
            trial_id = os.path.basename(trial_folder).split("_")[-1]
            if int(trial_id) < 0: continue
            results[trial_id] = {
                "actor_data": read_actor_data(trial_folder),
                "status_data": read_status_data(trial_folder)[0],
                "last_status_message": read_status_data(trial_folder)[1],
                "finish_time": read_status_data(trial_folder)[2],
            }
        except Exception:
            continue
    return results

# ── Config & Data Loading ──────────────────────────────────────────────────
scene_name = "crossing_0"
scene_results = process_scene(scene_name)

track_dict = {}
pair_file = f"/root/NonHoloPeopleAsPlanner/sim_ws/src/data_play/temp/{scene_name}/pair/data_id_label_pairs.txt"
if os.path.exists(pair_file):
    with open(pair_file, "r") as file:
        next(file)
        for line in file:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                track_dict[int(parts[0])] = parts[1]

config_path = f'/root/NonHoloPeopleAsPlanner/sim_ws/src/data_play/dataset/scene_config_30/{scene_name}.json'
with open(config_path, 'r') as f:
    scene_config = json.load(f)

obs_list = [obstacle(np.array(obs)) for obs in scene_config["obstacles"]]
robot_pos_goal = np.array(scene_config["robot_start_end"])
start_pos, goal_pos = robot_pos_goal[0:2], robot_pos_goal[2:4]

# ── CSV Logging ────────────────────────────────────────────────────────────
file_path = f"/root/test_data/{scene_name}/experiment_log.csv"
with open(file_path, "w") as f:
    f.write("Trial_ID, Finish, Total_collision, traveling_dis, Total_Time\n")

sorted_trials = sorted(scene_results.items(), key=lambda x: int(x[0]))
for trial_id, data in sorted_trials:
    status = data["status_data"]
    agent_list = [agent(aid, arr, track_dict) for aid, arr in data["actor_data"].items()]
    h_coll, o_coll, dist, prev_pos = 0, 0, 0, None
    
    for index, row in enumerate(status):
        curr_pos = row[1:3]
        if prev_pos is not None:
            dist += np.linalg.norm(curr_pos - prev_pos)
        prev_pos = curr_pos
        h_coll += sum(1 for ped in agent_list if ped.cal_index_collision(index, curr_pos))
        o_coll += sum(1 for obs in obs_list if obs.collsion(curr_pos))

    with open(file_path, "a") as f:
        f.write(f"{trial_id},{1 if data['finish_time'] else 0},{h_coll+o_coll},{dist},{data['finish_time']}\n")

# ── Plotting Separated Graphs ──────────────────────────────────────────────
latest_trial_id, latest_data = sorted_trials[-2]
status = latest_data["status_data"]
xs, ys, yaws = status[:, 1], status[:, 2], status[:, 3]
has_yaw = not np.all(np.isnan(yaws))

dx, dy = np.diff(xs), np.diff(ys)
norms = np.sqrt(dx**2 + dy**2)
norms[norms < 1e-9] = 1e-9
dx_n, dy_n = dx / norms, dy / norms

def setup_ax(ax, title):
    ax.set_aspect('equal')
    ax.set_title(title)
    for obs in obs_list:
        ax.fill(*obs.polygon.exterior.xy, color='black', alpha=0.8, zorder=2)
    ax.plot(xs, ys, color='royalblue', lw=1, alpha=0.4, zorder=3)
    ax.scatter(*start_pos, color='green', marker='^', s=150, zorder=7, label='Start')
    ax.scatter(*goal_pos, color='red', marker='*', s=150, zorder=7, label='Goal')
    ax.grid(True, alpha=0.2)

arrow_every = max(1, len(xs) // 25)
arrow_len = 0.5

# Plot 1: Velocity
fig1, ax1 = plt.subplots(figsize=(10, 10))
setup_ax(ax1, f"Trial {latest_trial_id}: Velocity Direction (Movement Path)")
for i in range(0, len(dx), arrow_every):
    ax1.annotate("", xy=(xs[i]+dx_n[i]*arrow_len, ys[i]+dy_n[i]*arrow_len), xytext=(xs[i], ys[i]),
                 arrowprops=dict(arrowstyle="-|>", color='limegreen', lw=1.5))
fig1.savefig(f"/root/test_data/{scene_name}/robot_velocity.png", dpi=150)

# Plot 2: Heading
if has_yaw:
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    setup_ax(ax2, f"Trial {latest_trial_id}: Robot Heading (Yaw)")
    for i in range(0, len(xs), arrow_every):
        if not np.isnan(yaws[i]):
            ax2.annotate("", xy=(xs[i]+np.cos(yaws[i])*arrow_len, ys[i]+np.sin(yaws[i])*arrow_len), 
                         xytext=(xs[i], ys[i]), arrowprops=dict(arrowstyle="-|>", color='darkorange', lw=1.5))
    fig2.savefig(f"/root/test_data/{scene_name}/robot_heading.png", dpi=150)

print(f"Done. Graphs saved to /root/test_data/{scene_name}/")