#!/usr/bin/env python3

import rospy
import time

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Quaternion
from std_msgs.msg import Int32
import json
import threading
import numpy as np
from sensor_msgs.msg import LaserScan
from data_play.msg import ModelInfo
import math
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Int32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import tf.transformations

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

from check_visibility import human_scoring, get_visible_region


class social_force:
    def __init__(self, obs_list, radius):
        self.obs_list = obs_list
        self.A = 1.5
        self.B = 1.0
        self.radius = radius
        self.v_pref = 1.0
        self.KI = 1.0

    def point_to_line_distance_with_point(self, x0, y0, x1, y1, x2, y2):
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        distance = abs(A * x0 + B * y0 + C) / math.sqrt(A**2 + B**2)

        if A**2 + B**2 == 0:
            x_closest = x0
            y_closest = y0
            return distance, x_closest, y_closest

        x_closest = (B * (B * x0 - A * y0) - A * C) / (A**2 + B**2)
        y_closest = (A * (-B * x0 + A * y0) - B * C) / (A**2 + B**2)

        dist1 = (x_closest - x1)**2 + (y_closest - y1)**2
        dist2 = (x_closest - x2)**2 + (y_closest - y2)**2
        dist_1_2 = (x1 - x2)**2 + (y1 - y2)**2

        if dist1 > dist_1_2 or dist2 > dist_1_2:
            x_closest = 100000
            y_closest = 100000
            distance = 100000

        return distance, x_closest, y_closest

    def predict(self, goal, state, humans, mapping, cv_pref):
        self.v_pref = cv_pref
        delta_position = goal - state[0:2]
        delta_x = delta_position[0]
        delta_y = delta_position[1]
        dist_to_goal = np.linalg.norm(delta_position)
        desired_vx = (delta_x / dist_to_goal) * self.v_pref
        desired_vy = (delta_y / dist_to_goal) * self.v_pref
        curr_delta_vx = self.KI * (desired_vx - state[2])
        curr_delta_vy = self.KI * (desired_vy - state[3])
        A = self.A
        B = self.B
        interaction_vx = 0
        interaction_vy = 0
        min_dist_to_human = 10000
        min_dist_to_obs = 10000
        vx_human = 0
        vy_human = 0
        vx_obs = 0
        vy_obs = 0

        my_position = state[0:2]
        min_dist_to_human = float('inf')

        for human in humans:
            human_id = int(human[0])
            other_human_pos = human[1:3]

            human_radius = 1

            if human_id not in mapping:
                continue
            if mapping[human_id] == 0:
                human_radius = 0.5
            elif mapping[human_id] == 1:
                human_radius = 0.5
            elif mapping[human_id] == 2:
                human_radius = 0.5

            delta_x = my_position[0] - other_human_pos[0]
            delta_y = my_position[1] - other_human_pos[1]
            dist_to_human = np.sqrt(delta_x**2 + delta_y**2)

            if dist_to_human == 0:
                continue

            if min_dist_to_human > dist_to_human:
                min_dist_to_human = dist_to_human
                vx_human = A * np.exp((self.radius + human_radius - dist_to_human) / B) * (delta_x / dist_to_human)
                vy_human = A * np.exp((self.radius + human_radius - dist_to_human) / B) * (delta_y / dist_to_human)

        interaction_vx += vx_human
        interaction_vy += vy_human

        for obstacles in self.obs_list:
            if obstacles.shape[0] == 1:
                delta_x = my_position[0] - obstacles[0, 0]
                delta_y = my_position[1] - obstacles[0, 1]
                dist_to_obs = np.sqrt(delta_x**2 + delta_y**2)
                if min_dist_to_obs > dist_to_obs:
                    min_dist_to_obs = dist_to_obs
                    vx_obs = 3 * A * np.exp((self.radius - dist_to_obs) / B) * (delta_x / dist_to_obs)
                    vy_obs = 3 * A * np.exp((self.radius - dist_to_obs) / B) * (delta_y / dist_to_obs)
            else:
                for row_index in range(0, obstacles.shape[0] - 1):
                    x0 = my_position[0]
                    y0 = my_position[1]
                    x1 = obstacles[row_index, 0]
                    y1 = obstacles[row_index, 1]
                    x2 = obstacles[row_index + 1, 0]
                    y2 = obstacles[row_index + 1, 1]
                    dist_to_obs, x_closest, y_closest = self.point_to_line_distance_with_point(x0, y0, x1, y1, x2, y2)
                    delta_x = my_position[0] - x_closest
                    delta_y = my_position[1] - y_closest
                    dist_to_obs = np.sqrt(delta_x**2 + delta_y**2)
                    if min_dist_to_obs > dist_to_obs:
                        min_dist_to_obs = dist_to_obs
                        vx_obs = 3 * A * np.exp((self.radius - dist_to_obs) / B) * (delta_x / dist_to_obs)
                        vy_obs = 3 * A * np.exp((self.radius - dist_to_obs) / B) * (delta_y / dist_to_obs)

        interaction_vx += vx_obs * 0.1
        interaction_vy += vy_obs * 0.1
        total_delta_vx = curr_delta_vx + interaction_vx
        total_delta_vy = curr_delta_vy + interaction_vy
        new_vx = state[2] + total_delta_vx
        new_vy = state[3] + total_delta_vy
        act_norm = np.linalg.norm([new_vx, new_vy])
        if act_norm > self.v_pref:
            return np.array([new_vx / act_norm * self.v_pref, new_vy / act_norm * self.v_pref])
        else:
            return np.array([new_vx, new_vy])


class OurPlanner:
    def __init__(self, goal, obs_list):

        self.default = False
        self.robot_range = 10
        self.robot_radius = 1.0
        self.previous_leader = None

        #################
        # Tunable Param #-----------------------------------
        #################
        # agent
        self.human_radius = 0.8
        self.human_speed_ideal = 1.4
        self.human_speed_min = 0.6
        self.robot_speed_max = 1.4

        # score function
        self.goal_score_steps = 10
        self.catchup_threshold = 1.5
        self.catchup_speed = 2.0
        self.position_penalty = -1.75

        # weight
        self.weight_goal = 1.0
        self.weight_velocity = 0.5
        self.weight_position = 1.0
        # FIX: Reduced alignment weight from 0.8 → 0.3 and threshold from 0.7 → 0.0
        # The original 0.8 weight was too dominant — it would push any human whose
        # heading didn't closely match the robot's current yaw below zero, causing
        # the planner to fall back to default even when good leaders existed.
        # Threshold 0.7 (cos 45°) was also too strict for a non-holonomic robot
        # that may not yet be facing the same direction as a valid leader.
        self.weight_alignment = 0.3
        self.alignment_cone_threshold = 0.0  # any forward-facing human counts
        self.current_leader_bias = 0.05

        # group identification
        self.min_distance_threshold = 1.5
        self.distance_threshold = 2.0
        self.velocity_threshold = 0.5

        # visibility
        self.inflate_radius = 0.5
        #-----------------------------------------------------

        self.goal = goal
        self.obs_list = obs_list
        self.history_list = []
        self.base_controller = social_force(self.obs_list, self.robot_radius)
        self.list_length = 25
        self.state_buffer = []
        self.human_buffer = []
        self.time_step = 0.11

    def predict(self, state, human_step, mask, laser_scan, robot_yaw=0.0):

        while len(self.state_buffer) < self.list_length:
            self.state_buffer.append(state)
        while len(self.human_buffer) < self.list_length:
            self.human_buffer.append(human_step)

        # print("NEW HUMAN STEP: ", human_step)
        # print(len(self.state_buffer), len(self.human_buffer), self.list_length)

        global_goal_x = self.goal[0]
        global_goal_y = self.goal[1]
        robot_goal_vector = (global_goal_x - state[0], global_goal_y - state[1])
        goal_distance = math.sqrt(robot_goal_vector[0]**2 + robot_goal_vector[1]**2)

        #########################
        # Leader Identification #
        #########################
        neighbor_traj = {}
        for timestep in self.human_buffer:
            # print("Timestep: ", timestep)
            for human in timestep:
                # print("Human: ", human)
                id, px, py, vx, vy = human
                if any(agent[0] == id for agent in human_step):
                    if id not in neighbor_traj:
                        neighbor_traj[id] = []
                    neighbor_traj[id].append([px, py, vx, vy])

        ########################
        # Group Identification #
        ########################
        groups = []
        visited = set()

        for i, human_a in enumerate(human_step):
            if human_a[0] in visited:
                continue
            group = [human_a]
            visited.add(human_a[0])

            for j, human_b in enumerate(human_step):
                if human_b[0] in visited or human_a[0] == human_b[0]:
                    continue
                distance = math.sqrt((human_a[1] - human_b[1])**2 + (human_a[2] - human_b[2])**2)
                velocity_similarity = math.sqrt((human_a[3] - human_b[3])**2 + (human_a[4] - human_b[4])**2)
                if distance <= self.distance_threshold and velocity_similarity <= self.velocity_threshold:
                    group.append(human_b)
                    visited.add(human_b[0])

            if len(group) > 1:
                groups.append(group)

        ####################
        # Visibility Check #
        ####################
        robot_pose = [state[0], state[1], robot_yaw]
        human_radius = [self.inflate_radius for i in range(len(human_step))]
        human_scores_list = human_scoring(laser_scan, human_step, robot_pose, human_radius)
        visible_region_edges = np.array([])

        invisible_list = []
        for k, score in enumerate(human_scores_list):
            if score < 0:
                invisible_list.append(human_step[k][0])

        # ── DEBUG: pipeline population ────────────────────────────────────────
        print(f"\n[DEBUG] robot yaw={math.degrees(robot_yaw):.1f}deg  "
              f"goal_dist={goal_distance:.2f}m  "
              f"goal_dir={math.degrees(math.atan2(robot_goal_vector[1], robot_goal_vector[0])):.1f}deg")
        # print(f"[DEBUG] human_step={len(human_step)}  "
        #       f"neighbor_traj={len(neighbor_traj)}  "
        #       f"invisible={invisible_list}")
        # ─────────────────────────────────────────────────────────────────────

        ###########
        # 1. Goal #
        ###########
        scores_goal = {}

        for ped_id, trajectory in neighbor_traj.items():
            num_steps = min(self.goal_score_steps, len(trajectory))
            avg_heading_vector = [0, 0]
            goal_vector = (global_goal_x - trajectory[-1][0], global_goal_y - trajectory[-1][1])
            for i in range(-1, -1 - num_steps, -1):
                avg_heading_vector[0] += trajectory[i][2]
                avg_heading_vector[1] += trajectory[i][3]
            avg_heading_magnitude = math.sqrt(avg_heading_vector[0]**2 + avg_heading_vector[1]**2)
            goal_magnitude = math.sqrt(goal_vector[0]**2 + goal_vector[1]**2)

            if avg_heading_magnitude > 0 and goal_magnitude > 0:
                avg_heading_vector = (avg_heading_vector[0] / avg_heading_magnitude, avg_heading_vector[1] / avg_heading_magnitude)
                goal_vector = (goal_vector[0] / goal_magnitude, goal_vector[1] / goal_magnitude)
                dot_product = avg_heading_vector[0] * goal_vector[0] + avg_heading_vector[1] * goal_vector[1]
                scores_goal[ped_id] = dot_product if dot_product >= 0.5 else -10
            else:
                scores_goal[ped_id] = -10

        ###############
        # 2. Velocity #
        ###############
        scores_velocity = {}
        human_speeds = {}
        ideal_speed = self.human_speed_ideal
        for ped_id, trajectory in neighbor_traj.items():
            total_distance = 0
            steps = min(10, len(trajectory))
            for i in range(-1, -1 - steps, -1):
                total_distance += math.sqrt(trajectory[i][2]**2 + trajectory[i][3]**2)
            avg_speed = total_distance / steps
            human_speeds[ped_id] = math.sqrt(trajectory[-1][2]**2 + trajectory[-1][3]**2)

            if avg_speed < self.human_speed_min:
                scores_velocity[ped_id] = (avg_speed - ideal_speed) / ideal_speed
                if avg_speed < 0.1:
                    scores_velocity[ped_id] = -10
            else:
                scores_velocity[ped_id] = max(0, 1 - (abs(avg_speed - ideal_speed) / ideal_speed))

        ###############
        # 3. Position #
        ###############
        scores_position = {}
        hr_distance = {}
        hr_vector = {}
        for ped_id, trajectory in neighbor_traj.items():
            human_pos = trajectory[-1][:2]
            human_vector = (human_pos[0] - state[0], human_pos[1] - state[1])
            distance = math.sqrt(human_vector[0]**2 + human_vector[1]**2)
            hr_distance[ped_id] = distance
            hr_vector[ped_id] = human_vector
            human_vector = (human_vector[0] / distance, human_vector[1] / distance)
            robot_heading = robot_goal_vector

            r_heading_mag = math.sqrt(robot_heading[0]**2 + robot_heading[1]**2)
            robot_heading = (robot_heading[0] / r_heading_mag, robot_heading[1] / r_heading_mag)

            dot_product = human_vector[0] * robot_heading[0] + human_vector[1] * robot_heading[1]
            if dot_product >= 0.5:
                scores_position[ped_id] = (dot_product + max(0, 1 - (distance / self.robot_range))) / 2
            else:
                scores_position[ped_id] = self.position_penalty

        #####################
        # 4. Alignment Cone #
        #####################
        scores_alignment = {}
        for ped_id, trajectory in neighbor_traj.items():
            h_vx, h_vy = trajectory[-1][2], trajectory[-1][3]
            h_mag = math.sqrt(h_vx**2 + h_vy**2)
            r_vx, r_vy = math.cos(robot_yaw), math.sin(robot_yaw)

            if h_mag > 0.1:
                alignment_dot = (h_vx / h_mag) * r_vx + (h_vy / h_mag) * r_vy
                # FIX: threshold 0.0 means any human moving in a generally
                # forward direction relative to the robot contributes positively.
                scores_alignment[ped_id] = alignment_dot if alignment_dot > self.alignment_cone_threshold else 0
            else:
                scores_alignment[ped_id] = 0

        # ── DEBUG: per-score breakdown before weighting ───────────────────────
        # print(f"[DEBUG] scores_goal:      {scores_goal}")
        # print(f"[DEBUG] scores_velocity:  {scores_velocity}")
        # print(f"[DEBUG] scores_position:  {scores_position}")
        # print(f"[DEBUG] scores_alignment: {scores_alignment}")
        # ─────────────────────────────────────────────────────────────────────

        #########
        # Total #
        #########
        total_scores = {}
        print("\n--- Leader Selection Scoring ---")
        print(f"{'ID':<6} | {'Goal':>6} | {'Vel':>6} | {'Pos':>6} | {'Align':>6} | {'TOTAL':>6}")
        print("-" * 55)

        for ped_id in neighbor_traj.keys():
            if ped_id in invisible_list:
                continue

            s_goal  = scores_goal.get(ped_id, 0)
            s_vel   = scores_velocity.get(ped_id, 0)
            s_pos   = scores_position.get(ped_id, 0)
            s_align = scores_alignment.get(ped_id, 0)

            weighted_score = (self.weight_goal      * s_goal  +
                              self.weight_velocity   * s_vel   +
                              self.weight_position   * s_pos   +
                              self.weight_alignment  * s_align)

            if ped_id == self.previous_leader:
                weighted_score += self.current_leader_bias
                label = f"*{ped_id}"
            else:
                label = f" {ped_id}"

            total_scores[ped_id] = weighted_score
            print(f"{label:<6} | {s_goal:>6.2f} | {s_vel:>6.2f} | {s_pos:>6.2f} | {s_align:>6.2f} | {weighted_score:>6.2f}")

        # ── DEBUG: why we fall back ───────────────────────────────────────────
        if not total_scores:
            print("[DEBUG] FALLBACK: no candidates in total_scores (all invisible or neighbor_traj empty)")
        elif not any(score > 0 for score in total_scores.values()):
            best = max(total_scores, key=total_scores.get)
            print(f"[DEBUG] FALLBACK: best candidate {best} scored {total_scores[best]:.2f} — no score > 0")
        elif goal_distance <= 1.0:
            print(f"[DEBUG] FALLBACK: within goal threshold ({goal_distance:.2f}m <= 1.0m)")
        # ─────────────────────────────────────────────────────────────────────

        if (
            total_scores
            and any(score > 0 for score in total_scores.values())
            and goal_distance > 1.0
        ):
            leader_ID = max(total_scores, key=total_scores.get)
            print(f"WINNER: Agent {leader_ID}  score={total_scores[leader_ID]:.2f}  dist={goal_distance:.2f}m")

            def get_closest_human_in_group(group, robot_x, robot_y):
                closest_human = None
                min_distance = float('inf')
                for human in group:
                    human_px, human_py = human[1], human[2]
                    distance = math.sqrt((robot_x - human_px)**2 + (robot_y - human_py)**2)
                    if distance < min_distance:
                        closest_human = human
                        min_distance = distance
                return closest_human

            leader_group = None
            for group in groups:
                if any(human[0] == leader_ID for human in group):
                    leader_group = group
                    break
            if leader_group:
                closest_human = get_closest_human_in_group(leader_group, state[0], state[1])
                if (
                    closest_human[0] != leader_ID
                    and closest_human[0] not in invisible_list
                    and total_scores[closest_human[0]] > 0
                ):
                    leader_ID = closest_human[0]
                    print(f"Switch to closest human in group: {leader_ID}")

            self.following_id_vis = leader_ID
            self.previous_leader = leader_ID

            ###############
            # Set Subgoal #
            ###############
            vx = hr_vector[leader_ID][0]
            vy = hr_vector[leader_ID][1]

            base_gx = neighbor_traj[leader_ID][-1][0]
            base_gy = neighbor_traj[leader_ID][-1][1]

            def get_candidate_positions(vx, vy, angle_range=90):
                positions = []
                magnitude = (vx**2 + vy**2)**0.5
                if magnitude != 0:
                    for angle in range(-angle_range, angle_range + 1, 2):
                        angle_rad = math.radians(angle)
                        rotated_vx = vx * math.cos(angle_rad) - vy * math.sin(angle_rad)
                        rotated_vy = vx * math.sin(angle_rad) + vy * math.cos(angle_rad)
                        pos_gx = base_gx - self.human_radius * rotated_vx / magnitude
                        pos_gy = base_gy - self.human_radius * rotated_vy / magnitude
                        positions.append((pos_gx, pos_gy))
                else:
                    positions.append((base_gx, base_gy))
                return positions

            def evaluate_position(goal_x, goal_y):
                min_distance = float('inf')
                for human in human_step:
                    if human[0] == leader_ID:
                        continue
                    human_px, human_py = human[1], human[2]
                    distance = math.sqrt((human_px - goal_x)**2 + (human_py - goal_y)**2)
                    min_distance = min(min_distance, distance)
                return min_distance

            candidate_positions = get_candidate_positions(vx, vy)
            best_position = max(candidate_positions, key=lambda pos: evaluate_position(pos[0], pos[1]))
            new_gx, new_gy = best_position

            min_safe_distance = 2.0
            min_distance = evaluate_position(new_gx, new_gy)
            if min_distance < min_safe_distance:
                print('##### Push further ######')
                direction_vx = new_gx - base_gx
                direction_vy = new_gy - base_gy
                direction_mag = (direction_vx**2 + direction_vy**2)**0.5
                if direction_mag > 0:
                    scale = (min_safe_distance / min_distance) * 1.0
                    new_gx = base_gx + scale * direction_vx
                    new_gy = base_gy + scale * direction_vy

            #################
            # Set New Speed #
            #################
            command_v_pref = 0
            if hr_distance[leader_ID] > self.catchup_threshold:
                command_v_pref = self.catchup_speed
                print("---catching up---")
            else:
                command_v_pref = human_speeds[leader_ID]

        else:
            print(f"Back to default planner   distance: {goal_distance:.2f}")
            self.following_id_vis = -1
            new_gx = global_goal_x
            new_gy = global_goal_y
            command_v_pref = self.robot_speed_max

        if self.default:
            print("##### Using default SF planner #####")
            new_gx = global_goal_x
            new_gy = global_goal_y
            command_v_pref = self.robot_speed_max

        comand_goal = np.array([new_gx, new_gy])
        action = self.base_controller.predict(comand_goal, state, human_step, mask, command_v_pref)

        try:
            self.state_buffer.pop(0)
            self.human_buffer.pop(0)
        except Exception as e:
            rospy.logwarn(f"Error while popping from buffers: {e}")

        return self.following_id_vis, action, comand_goal, invisible_list, visible_region_edges

    def clear_buffer(self):
        self.state_buffer = []
        self.human_buffer = []


class Robot:

    def __init__(self):
        self.lock = threading.Lock()
        rospy.init_node('robot_listener', anonymous=True)
        scene = rospy.get_param("scene", "nexus_2_0")
        self.robot_horizon = 15.0
        self.scene_path = '/root/NonHoloPeopleAsPlanner/sim_ws/src/data_play/dataset/scene_config_30/' + scene + '.json'

        with open(self.scene_path, 'r') as f:
            self.scene_config = json.load(f)

        robot_pos_goal = np.array(self.scene_config["robot_start_end"])

        self.start_pos = robot_pos_goal[0:2]
        self.goal_pos = robot_pos_goal[2:4]
        self.gaol_range = 0.3

        self.obs_list = []
        for obs in self.scene_config["obstacles"]:
            self.obs_list.append(np.array(obs))

        self.planner = OurPlanner(self.goal_pos, self.obs_list)

        self.start_mission = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.robot_state = None
        self.id_mask = None
        self.visable_humans = None
        self.laser_point = None
        self.laser_scan = []

        # Subscribers
        self.model_info_sub = rospy.Subscriber('/gazebo/model_info', ModelInfo, self.model_info_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.mission_sub = rospy.Subscriber('/env_control', Int32, self.mission_callback)
        self.laser_sub = rospy.Subscriber('/robot_1/laser_scan', LaserScan, self.laser_callback)
        self.invisable_pub = rospy.Publisher('/invisable_id', Int32MultiArray, queue_size=1)
        self.vis_pub = rospy.Publisher("/vis_array_topic", Float32MultiArray, queue_size=10)

        # Publisher
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        rospy.loginfo("Robot Node Initialized: Subscribed to /gazebo/model_info and /odom, Publishing to /cmd_vel")

        self.edges_visible_region = np.array([])
        self.pubVisibleEdges = rospy.Publisher('/visibleEdges', Marker, queue_size=2)

    def pubVisibleRegion(self):
        marker = Marker()
        marker.header.frame_id = "robot_1/base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "edges"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = []
        for i in range(len(self.edges_visible_region)):
            point1, point2 = self.edges_visible_region[i]
            start_point = Point(point1[0], point1[1], 0)
            end_point = Point(point2[0], point2[1], 0)
            marker.points.append(start_point)
            marker.points.append(end_point)
        self.pubVisibleEdges.publish(marker)

    def laser_callback(self, msg):
        self.laser_scan = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.range_max = msg.range_max

    def _reset_robot_pose(self):
        model_state = ModelState()
        model_state.model_name = 'robot_1'
        model_state.pose.position.x = self.start_pos[0]
        model_state.pose.position.y = self.start_pos[1]
        model_state.pose.position.z = 0.0
        model_state.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.set_state(model_state)

    def mission_callback(self, msg):
        if msg.data == 0:
            self._reset_robot_pose()
            self.planner.clear_buffer()
        with self.lock:
            self.start_mission = msg.data

    def model_info_callback(self, msg):
        # print("we in the model info callback", self.id_mask, self.robot_state, msg)
        if self.id_mask is None:
            self.id_mask = {}
            for i in range(len(msg.ids)):
                model_id = msg.ids[i]
                model_tag = msg.tags[i]
                self.id_mask[model_id] = model_tag

        if self.robot_state is None:
            return

        with self.lock:
            state_now = self.robot_state[0:4]
            robot_yaw = self.robot_state[4] if len(self.robot_state) >= 5 else 0.0

        model_temp = []
        for i in range(len(msg.ids)):
            model_id = msg.ids[i]
            px, py = msg.position[i].x, msg.position[i].y
            vx, vy = msg.velocity[i].x, msg.velocity[i].y
            dx = px - state_now[0]
            dy = py - state_now[1]
            if np.sqrt(dx**2 + dy**2) < self.robot_horizon:
                model_temp.append([model_id, px, py, vx, vy])
        self.visable_humans = model_temp

        if self.start_mission == 1:

            # print("WE ARE IN THE MISSION")

            track_id, action, subgoal, invis_index, edges_visible_region = self.planner.predict(
                state_now, model_temp, self.id_mask, self.laser_scan, robot_yaw
            )

            if np.linalg.norm(self.goal_pos - state_now[0:2]) < self.gaol_range:
                action = np.array([0, 0])
                self.start_mission = 0
                self.planner.clear_buffer()

            cmd_msg = Twist()
            desired_vx = action[0]
            desired_vy = action[1]

            desired_yaw = math.atan2(desired_vy, desired_vx)
            yaw_error = (desired_yaw - robot_yaw + math.pi) % (2 * math.pi) - math.pi

            desired_speed = math.sqrt(desired_vx**2 + desired_vy**2)
            v_turn = desired_speed * max(0.0, math.cos(yaw_error))

            K_omega = 2.0
            omega = max(-1.0, min(1.0, K_omega * yaw_error))

            cmd_msg.linear.x = v_turn
            cmd_msg.linear.y = 0.0
            cmd_msg.angular.z = omega

            self.cmd_vel_pub.publish(cmd_msg)

            invis_id_msg = Int32MultiArray()
            invis_id_msg.data = invis_index
            self.invisable_pub.publish(invis_id_msg)

            vis_msg = Float32MultiArray()
            vis_msg.data = [track_id, subgoal[0], subgoal[1], self.robot_state[0], self.robot_state[1]]
            self.vis_pub.publish(vis_msg)

        else:
            invis_id_msg = Int32MultiArray()
            invis_id_msg.data = []
            self.invisable_pub.publish(invis_id_msg)

            vis_msg = Float32MultiArray()
            vis_msg.data = [-1, self.goal_pos[0], self.goal_pos[1], self.robot_state[0], self.robot_state[1]]
            self.vis_pub.publish(vis_msg)

            cmd_msg = Twist()
            cmd_msg.linear.x = 0.0
            cmd_msg.linear.y = 0.0
            cmd_msg.angular.z = 0.0
            self.cmd_vel_pub.publish(cmd_msg)

            self._reset_robot_pose()
            self.planner.clear_buffer()

    def odom_callback(self, msg):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(orientation_list)

        v_body = msg.twist.twist.linear.x
        vx = v_body * math.cos(yaw)
        vy = v_body * math.sin(yaw)

        with self.lock:
            self.robot_state = [px, py, vx, vy, yaw]


if __name__ == '__main__':
    try:
        robot = Robot()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass