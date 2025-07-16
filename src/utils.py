#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import (
    Point, Pose, PoseStamped, PoseArray,
    Quaternion, PolygonStamped, Polygon, Point32,
    PoseWithCovarianceStamped, PointStamped
)
import tf.transformations
import tf
import matplotlib.pyplot as plt
import time


class CircularArray(object):
    """ Simple implementation of a circular array.
        You can append to it any number of times but only "size" items will be kept
    """
    def __init__(self, size):
        self.arr = np.zeros(size)
        self.ind = 0
        self.num_els = 0

    def append(self, value):
        if self.num_els < self.arr.shape[0]:
            self.num_els += 1
        self.arr[self.ind] = value
        self.ind = (self.ind + 1) % self.arr.shape[0]

    def mean(self):
        return np.mean(self.arr[:self.num_els])

    def median(self):
        return np.median(self.arr[:self.num_els])


class Timer:
    """ Simple helper class to compute the rate at which something is called.
        
        "smoothing" determines the size of the underlying circular array, which averages
        out variations in call rate over time.

        use timer.tick() to record an event
        use timer.fps() to report the average event rate.
    """
    def __init__(self, smoothing):
        self.arr = CircularArray(smoothing)
        self.last_time = time.time()

    def tick(self):
        t = time.time()
        self.arr.append(1.0 / (t - self.last_time))
        self.last_time = t

    def fps(self):
        return self.arr.mean()


def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    x, y, z, w = tf.transformations.quaternion_from_euler(0, 0, angle)
    return Quaternion(x=x, y=y, z=z, w=w)


def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians (yaw)."""
    roll, pitch, yaw = tf.transformations.euler_from_quaternion(
        (q.x, q.y, q.z, q.w)
    )
    return yaw


def rotation_matrix(theta):
    """Creates a 2×2 rotation matrix for the given angle in radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def particle_to_pose(particle):
    """Converts a particle [x, y, theta] into a ROS Pose message."""
    pose = Pose()
    pose.position.x = float(particle[0])
    pose.position.y = float(particle[1])
    pose.orientation = angle_to_quaternion(particle[2])
    return pose


def particles_to_poses(particles):
    """Converts an array of particles into a list of Pose messages."""
    return [particle_to_pose(p) for p in particles]


def make_header(frame_id, stamp=None):
    """Creates a ROS Header for stamped messages."""
    if stamp is None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def map_to_world(poses, map_info):
    """
    Vectorized conversion from map (pixels) to world (meters).
    Modifies poses in place: [[x,y,theta],...].
    """
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)
    c, s = np.cos(angle), np.sin(angle)
    # rotate
    x = poses[:, 0].copy()
    poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
    poses[:, 1] = s * x + c * poses[:, 1]
    # scale and translate
    poses[:, :2] *= float(scale)
    poses[:, 0] += map_info.origin.position.x
    poses[:, 1] += map_info.origin.position.y
    poses[:, 2] += angle


def world_to_map(poses, map_info):
    """
    Vectorized conversion from world (meters) to map (pixels).
    Modifies poses in place: [[x,y,theta],...].
    """
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)
    # translate and scale
    poses[:, 0] -= map_info.origin.position.x
    poses[:, 1] -= map_info.origin.position.y
    poses[:, :2] *= (1.0 / float(scale))
    # rotate
    c, s = np.cos(angle), np.sin(angle)
    x = poses[:, 0].copy()
    poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
    poses[:, 1] = s * x + c * poses[:, 1]
    poses[:, 2] += angle
