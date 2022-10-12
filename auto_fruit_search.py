# M4 - Autonomous fruit searching

# basic python packages
from math import sqrt
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

#import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure



def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 30 # tick
    
    # turn towards the waypoint
    #Calculate the angle required to turn
    dif_y = waypoint[1]-robot_pose[1]
    dif_x = waypoint[0]-robot_pose[0]
    #turn_angle = abs(np.arctan(dif_y,dif_x)-robot_pose[-1])
    turn_angle = np.arctan2(dif_y,dif_x)-robot_pose[2]

    turn_time = turn_angle*baseline/(wheel_vel*scale) #replace with your calculation
    #Charlie - check line above for equation
    print("Turning for {:.2f} seconds".format(turn_time))
    #Get the robots pose
    l_vl,r_vl = ppi.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)

    #Creating the array for raw measurements
    raw_meas = np.array([l_vl,r_vl,turn_time])
    robot_pose = get_robot_pose(ekf,raw_meas)

    # after turning, drive straight to the waypoint
    dist_to_point = np.sqrt((waypoint[0]-robot_pose[0])**2+(waypoint[1]-robot_pose[1])**2)

    drive_time = dist_to_point/(scale*wheel_vel) # replace with your calculation
    print("Driving for {:.2f} seconds".format(drive_time))
    l_vl,r_vl = ppi.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    
    ####################################################
    raw_meas = np.array([l_vl,r_vl,turn_time])
    robot_pose= get_robot_pose(ekf,raw_meas)

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose(ekf,drive_meas):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    #Using SLAM to get the robots position
    ekf.predict(drive_meas)
    lms ,_ = aruco_det.detect_marker_positions(ppi.get_image())
    ekf.update(lms)
    robot_pose = ekf.get_state_vector()
    ####################################################

    return robot_pose