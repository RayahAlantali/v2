#!/usr/bin/env python
import os
import sys
import math
import random
import numpy as np
from math import pi

import copy
import rospy
import tf
import message_filters
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import DeleteModel, SpawnModel
from gazebo_msgs.srv import SetModelState, GetModelState, GetLinkState
from geometry_msgs.msg import *
from geometry_msgs.msg import Point, Quaternion
from tqdm import tqdm
import csv


class DataCollector:
    def __init__(self, args):
        # place to hide all objects
        self.obj_bin = [0, 0, 3.5, 0, 0, 0]
        self.args = args
        self.bridge = CvBridge()
        self.model_list = self.spawn_all_objs()
        # TODO: change hardcoded camera matrix
        self.K = np.array([584.4871239733801076, 0.0, 335.6960535832945425,
                           0.0, 598.8642626421145678, 288.7116086147667602,
                           0.0, 0.0, 1.0]).reshape(3, 3)
        C_gazebo = self.get_link_pose('pibot_cam_link', 'world', 'se3')
        gazebo2cv = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
            ])
        self.cam_pose = C_gazebo.dot(gazebo2cv)
        
        self.img = None
        self.depth = None
        img_sub = message_filters.Subscriber("/pibot_cam/rgb/image_raw", Image)
        depth_sub = message_filters.Subscriber("/pibot_cam/depth/image_raw", Image)
        ts = message_filters.ApproximateTimeSynchronizer(
            [img_sub, depth_sub], 10, 0.01)
        ts.registerCallback(self.ts_callback)
        
    def collect(self):
        rospy.sleep(2)
        for label, sub_model_list in self.model_list.items():
            sample_per_model = int(self.args.sample_per_class/len(sub_model_list))
            depth_thresh = 2
            for model in sub_model_list:
                rgb_path = os.path.join('dataset', self.args.dataset_name, model, 'images')
                label_path = os.path.join('dataset', self.args.dataset_name, model, 'labels')
                label_path_readable = os.path.join('dataset', self.args.dataset_name, model, 'labels_readable')
                if not os.path.exists(rgb_path):
                    os.makedirs(rgb_path)
                if not os.path.exists(label_path):
                    os.makedirs(label_path)
                if not os.path.exists(label_path_readable):
                    os.makedirs(label_path_readable)  
                for i in tqdm(range(sample_per_model),  desc=model):
                    self.shuffle_model(model, z_low=0.3, z_high=1.5)
                    rospy.sleep(0.1)
                    if self.img is not None and self.depth is not None:
                        sample_name =  "%s_%i" % (model, i)
                        label_image = np.where(self.depth < depth_thresh, label, 0).astype(np.uint8)
                        label_image_readable = np.where(self.depth < depth_thresh, label*75, 0).astype(np.uint8)
                        if len(np.unique(label_image)) > 1:
                            cv2.imwrite(os.path.join(rgb_path, sample_name+".jpg"), self.img)
                            cv2.imwrite(os.path.join(label_path, sample_name+"_label.png"), label_image)
                            cv2.imwrite(os.path.join(label_path_readable, sample_name+"_label_readable.png"), label_image_readable)
                self.hide_model(model)
    
    def ts_callback(self, img_msg, depth_msg):
        try:
            self.img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            self.depth = self.bridge.imgmsg_to_cv2(depth_msg)
        except CvBridgeError as e:
            print(e)

    #################################
    #           Tool Box            #
    #################################
            
    def spawn_all_objs(self):
        model_list = {}
        print("Waiting for gazebo services...")
        rospy.wait_for_service("gazebo/spawn_urdf_model")
        spawn_model = rospy.ServiceProxy("gazebo/spawn_urdf_model", SpawnModel)
        x, y, z = self.obj_bin[:3]
        roll, pitch, yaw = self.obj_bin[3:]
        chosen_classes = self.args.class_labels
        urdf_bank = self.get_urdf_bank()
        if self.args.class_labels == -1:
            chosen_classes = range(1, len(urdf_bank)+1)
        for label in chosen_classes:
            try:
                for instance in urdf_bank[label]:
                    try:
                        model_list[label].append(instance['name'])
                    except KeyError:
                        model_list[label] = []
                        model_list[label].append(instance['name'])
                    quat = np.array(tf.transformations.quaternion_from_euler(
                        roll, pitch, yaw))
                    target_pose = Pose(Point(x, y, z),Quaternion(
                        quat[0], quat[1], quat[2], quat[3]))
                    spawn_model(instance['name'], instance['urdf'], "",
                                target_pose, "world")
            except KeyError:
                print("requested class index %s doesnot exist" % label)
        return model_list

    def shuffle_model(self, model, z_low, z_high):
        rand_z = np.random.uniform(low=z_low, high=z_high)
        focal_lengh =  np.trace(self.K[:2, :2])/2
        border_crop = 30
        x_range = (self.K[0, 2]-border_crop)/focal_lengh * rand_z
        y_range = (self.K[1, 2] - border_crop)/focal_lengh * rand_z
        x = np.random.uniform(-x_range, x_range)
        y = np.random.uniform(-y_range, y_range)
        rand_mat = tf.transformations.random_rotation_matrix()
        rand_mat[:3, 3] = [x, y, rand_z]
        # transform to world coordinate
        target_pose = self.cam_pose.dot(rand_mat)            
        target_quat = tf.transformations.quaternion_from_matrix(rand_mat)
        self.move_model(model, target_pose[:3, 3], target_quat)         

    def hide_model(self, model):
        xyz = self.obj_bin[:3]
        roll, pitch, yaw = self.obj_bin[3:]
        quat =  np.array(
            tf.transformations.quaternion_from_euler(roll, pitch, yaw))
        self.move_model(model, xyz, quat)

    @staticmethod
    def move_model(model, xyz, quat, ref="world"):
        target = ModelState()
        target.model_name = model
        target.reference_frame = ref
        x, y, z = xyz
        target.pose.position = Point(x, y, z)
        target.pose.orientation = Quaternion(quat[0], quat[1],
                                             quat[2], quat[3])
        try:
            set_position = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)
            set_position(target)
        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)

    def get_urdf_bank(self):
        model_config_path = self.args.model_config_path
        if os.path.exists(model_config_path):
            model_f = open(model_config_path, 'r')
            reader = list(csv.DictReader(model_f))
            urdf_bank = {}
            for _, l in enumerate(reader):
                urdf_file_path = l['urdf_path']
                if os.path.exists(urdf_file_path):
                    with open(urdf_file_path) as f:
                        urdf_temp = f.read()
                    model_temp = {'name':l['name'], 'urdf':urdf_temp}
                    try:
                        urdf_bank[int(l['label'])].append(model_temp)
                    except KeyError:
                        urdf_bank[int(l['label'])] = []
                        urdf_bank[int(l['label'])].append(model_temp)
                else:
                    raise Exception("urdf file %s cannot be found" % urdf_file_path)
            return urdf_bank
        else:
            raise Exception("model config file does not exist")

    def delete_model(self, model):
        try:
            delete_model = rospy.ServiceProxy("gazebo/delete_model",
                                              DeleteModel)
            delete_model(model)
        except rospy.ServiceException as e:
            print ("Service call failed: %s" % e)
            

    def delete_all_objs(self):
        for _, var in self.model_list.items():
            for obj_name in var:
                self.delete_model(obj_name)
                print("model %s is deleted", obj_name)
                rospy.sleep(0.05)
            
    def get_link_pose(self, link_name, reference_frame, repr='se3'):
        rospy.wait_for_service('/gazebo/get_link_state')
        try:
            current_link_state = rospy.ServiceProxy(
                '/gazebo/get_link_state', GetLinkState)
            link_state = current_link_state(link_name, reference_frame)
            link_pose = link_state.link_state
            if repr == 'se3':
                return self.pose_msg2se3(link_pose.pose)
            elif repr == 'quat':
                return self.pose_msg2vec(link_pose.pose)
        except rospy.ServiceException as e:
            print ("Service call failed: %s", e)
     
    @staticmethod       
    def pose_msg2vec(pose_msg):
        pose_vec = np.array([pose_msg.position.x,
                            pose_msg.position.y,
                            pose_msg.position.z,
                            pose_msg.orientation.x,
                            pose_msg.orientation.y,
                            pose_msg.orientation.z,
                            pose_msg.orientation.w])
        return pose_vec

    @staticmethod
    def pose_msg2se3(pose_msg):
        t = np.array([pose_msg.position.x,
                    pose_msg.position.y, pose_msg.position.z])
        Q = np.array([pose_msg.orientation.x,
                        pose_msg.orientation.y,
                        pose_msg.orientation.z,
                        pose_msg.orientation.w])
        R = tf.transformations.quaternion_matrix(Q)
        I = tf.transformations.identity_matrix() # 4x4 Identity_matrix
        X = R+tf.transformations.translation_matrix(t)-I
        return X


if __name__ == '__main__':
    import argparse
    rospy.init_node("spawn_objects")
    #
    generator_parser = argparse.ArgumentParser(description='Generate a Dataset')
    #
    generator_parser.add_argument('--model_config_path', type=str, default="model_list.csv",
                                  help='dataset name')
    generator_parser.add_argument('--class_labels' , nargs='+', type=int, default=-1,
                                  help='start collecting data for given model idx in config file')
    generator_parser.add_argument('--dataset_name', type=str, default="test",
                                  help='dataset name')
    generator_parser.add_argument('--sample_per_class', type=int, default=100,
                                  help='number of samples per class')
    #
    args = generator_parser.parse_args()
    
    dataset_collector = DataCollector(args)
    dataset_collector.collect()
