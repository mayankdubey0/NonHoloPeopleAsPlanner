#!/usr/bin/env python3

import sys
import rospy
import math
import numpy as np
import copy
from typing import List, Dict, Set
import time

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped, PointStamped
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
import tf.transformations as tf
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32
from std_msgs.msg import Bool
from std_msgs.msg import ColorRGBA


class Controller:
    def __init__(self):
        self.has_get_initial_pose = False

        # FIX: Replace holonomic vx/vy with unicycle v (forward speed) and omega (yaw rate).
        # The old code stored vx and vy from cmd_vel.linear.x/.y and applied them
        # directly as world-frame displacements, which is holonomic behaviour.
        # A non-holonomic robot can only move forward along its heading, so we
        # store forward speed and angular rate instead.
        self.v = 0.0      # forward speed along robot heading (m/s)
        self.omega = 0.0  # yaw rate (rad/s)

        self.stop = False

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.callback_model_states)
        rospy.Subscriber('/stop', Bool, self.callback_stop)
        rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)

        self.pub_model_state = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=5)

    def set_move_step(self, step):
        self.step = step

    def set_control_rate(self, control_rate):
        self.control_rate = control_rate

    def set_robot_pose(self, pose: tuple):
        self.robot_pose = pose

    def get_robot_pose(self) -> tuple:
        return self.robot_pose

    def callback_model_states(self, data: ModelStates):
        for i, model_name in enumerate(data.name):
            if "robot_1" in model_name:
                robot_x = data.pose[i].position.x
                robot_y = data.pose[i].position.y
                orientation = data.pose[i].orientation
                quaternion = (orientation.x, orientation.y, orientation.z, orientation.w)
                euler = tf.euler_from_quaternion(quaternion)
                yaw = euler[2]
                self.set_robot_pose((robot_x, robot_y, yaw))
                if not self.has_get_initial_pose:
                    self.has_get_initial_pose = True

    def callback_stop(self, data: Bool):
        self.stop = data.data

    def cmd_vel_callback(self, msg):
        # FIX: Read forward speed from linear.x and yaw rate from angular.z.
        # Previously this read linear.x -> vx and linear.y -> vy and used them
        # as direct world-frame velocity components (holonomic). Now we treat
        # linear.x as the scalar forward speed and angular.z as the turning
        # rate, matching the unicycle commands published by our_robot.py.
        self.v = msg.linear.x
        self.omega = msg.angular.z
        print(f"(NH) Get velocity: v={self.v:.3f}, omega={self.omega:.3f}")

    def main_loop(self):
        self.rate = rospy.Rate(self.control_rate)
        self.dt = 1.0 / self.control_rate

        while not rospy.is_shutdown():
            if not self.has_get_initial_pose:
                print("Robot initial position has not been received!")
                self.rate.sleep()
                continue

            if not self.stop:
                curr_pose = self.get_robot_pose()
                curr_x, curr_y, curr_yaw = curr_pose

                # FIX: Unicycle kinematic integration.
                # Old code:
                #   new_x   = curr_x + vx * dt
                #   new_y   = curr_y + vy * dt
                #   new_yaw = curr_yaw  <- never updated, robot never rotated
                #
                # New code integrates the standard unicycle model:
                #   yaw += omega * dt
                #   x   += v * cos(yaw) * dt
                #   y   += v * sin(yaw) * dt
                #
                # Yaw is updated BEFORE projecting forward speed (semi-implicit
                # Euler) so that mid-step rotation is applied first, which gives
                # more stable integration at low control rates.
                new_yaw = curr_yaw + self.omega * self.dt
                new_x   = curr_x   + self.v * math.cos(new_yaw) * self.dt
                new_y   = curr_y   + self.v * math.sin(new_yaw) * self.dt

                new_pose = (new_x, new_y, new_yaw)
                self.set_robot_pose(new_pose)

                robot_state = ModelState()
                robot_state.model_name = "robot_1"

                # FIX: Publish the integrated yaw so Gazebo renders the robot
                # rotating visually. Previously vehicle_yaw = curr_pose[2] was
                # passed here, which never changed because yaw was never updated
                # -- the robot always faced the same direction in simulation.
                geo_quat = tf.quaternion_from_euler(0.0, 0.0, new_yaw)
                robot_state.pose.orientation = Quaternion(*geo_quat)

                robot_state.pose.position.x = new_x
                robot_state.pose.position.y = new_y
                robot_state.pose.position.z = 1
                self.pub_model_state.publish(robot_state)

            self.rate.sleep()


def main():
    rospy.init_node('robot_controller')

    control_rate = rospy.get_param('/control_rate', 30)
    step = rospy.get_param('/step', 0.2)

    robot_controller = Controller()
    robot_controller.set_move_step(step)
    robot_controller.set_control_rate(control_rate)

    rospy.sleep(2)
    robot_controller.main_loop()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass