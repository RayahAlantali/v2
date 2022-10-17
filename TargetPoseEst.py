# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
from os import listdir

import torch
from sklearn.cluster import KMeans

def get_bounding_box(fruit_select):
    #Multiplying the pixel values by the appropraite scale
    xmin = fruit_select[0] *480/240
    ymin = fruit_select[1] *640/320
    xmax = fruit_select[2] *480/240
    ymax = fruit_select[3] *640/240
    fruit = fruit_select[5]

    fruit_xcent = (xmin + xmax)/2
    fruit_ycent = (ymin + ymax)/2 
    fruit_width = xmax - xmin
    fruit_height = ymax - ymin

    class_converter = {0:1,1:3,2:4,3:5,4:2}

    return (class_converter[fruit], [fruit_xcent,fruit_ycent, fruit_width, fruit_height])

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses):
    # there are at most five types of targets in each image
    target_lst_box = [[], [], [], [], []]
    target_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = pear, 4 = orange, 5 = strawberry, 0 = not_a_target
    img_file_path = os.path.join(base_dir, file_path)
    model.conf = 0.2

    result = model(img_file_path)
    image_data = result.pandas().xyxy[0]
    image_data_list = image_data.values.tolist()

    yolo = []

    for fruit_data in image_data_list:
        yolo.append(get_bounding_box(fruit_data))

    all_vals = yolo

    for (target_num, box) in all_vals:
        pose = image_poses[file_path] # [x, y, theta] 
        target_lst_box[target_num-1].append(box) # bounding box of target
        target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []

    apple_dimensions = [0.075448, 0.074871, 0.071889]
    target_dimensions.append(apple_dimensions)
    lemon_dimensions = [0.060588, 0.059299, 0.053017]
    target_dimensions.append(lemon_dimensions)
    pear_dimensions = [0.0946, 0.0948, 0.135]
    target_dimensions.append(pear_dimensions)
    orange_dimensions = [0.0721, 0.0771, 0.0739]
    target_dimensions.append(orange_dimensions)
    strawberry_dimensions = [0.052, 0.0346, 0.0376]
    target_dimensions.append(strawberry_dimensions)

    target_list = ['apple', 'lemon', 'pear', 'orange', 'strawberry']

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]
        
        ######### Replace with your codes #########
    
        x_robot = robot_pose[0][0]
        y_robot = robot_pose[1][0]
        theta_robot = robot_pose[2][0]

        A = focal_length * true_height / box[3][0] # actual depth of object

        x_camera = box[0][0] / focal_length * A
        theta_camera = np.arctan(x_camera/A)
        dist_fruit = np.hypot(x_camera, theta_camera)
        theta_dif = theta_robot - theta_camera

        x_object = dist_fruit * np.cos(theta_dif)
        y_object = dist_fruit * np.cos(theta_dif)
        
        x_object_world = x_robot + x_object
        y_object_world = y_robot + y_object

        target_pose = {'y':y_object_world,'x':x_object_world}
        target_pose_dict[target_list[target_num-1]] = target_pose
         ###########################################
    
    return target_pose_dict

def merge_to_mean(position_est, remove_outlier = False):

    # Inputs:
    # position_est : An numpy array of coordinates {position_est[estimation #][0 = x, 1 = y]}
    # remove_outlier : Boolean (Remove outliers using Standard Distribution z-scores)
    # Outputs:
    # new_mean : An numpy array of coordinates {new_mean[0 = x, 1 = y]}

    # Check if the position_est has no elements
    if len(position_est) == 0:
        return None

    # Set up working parameters
    position_est_result = []
    z_threshold = 3

    # Compute mean and standard deviations
    means = np.mean(position_est, axis = 0)
    stds = np.std(position_est, axis = 0)
    mean_x = means[0]
    std_x = stds[0]
    mean_y = means[1]
    std_y = stds[1]
    
    # Remove outliers
    if remove_outlier:
        for i in range(len(position_est)):
            coordinates = position_est[i]
            z_score_x = (coordinates[0] - mean_x)/std_x
            z_score_y = (coordinates[1] - mean_y)/std_y
            if np.abs(z_score_x) > z_threshold or np.abs(z_score_y) > z_threshold:
                position_est_result.append(coordinates)
    else:
        position_est_result = position_est

    # Compute Mean
    new_mean = np.mean(position_est_result, axis = 0)

    return new_mean


def sort_locations_and_merge(position_est, distance_threshold = 0.3, remove_outlier = False, use_Kmeans = False):

    # Inputs:
    # position_est : An numpy array of coordinates {position_est[estimation #][0 = x, 1 = y]}
    # distance_threshold : the distance assumption that two fruits of the same type will be apart for
    # remove_outlier : Boolean (Remove outliers using Standard Distribution z-scores)
    # Outputs:
    # new_mean : An numpy array of coordinates {new_mean[0 = x, 1 = y]}

    # Initialize two sets of position estimations for each fruit of the same type
    position_est1 = []
    position_est2 = []

    # Sort data
    for i in range(len(position_est)):

        if(use_Kmeans):

            kmeans = KMeans(n_clusters = 2)
            kmeans.fit(position_est)
            if(kmeans.labels_[i] == 0):
                position_est1.append(position_est[i])
            else:
                position_est2.append(position_est[i])

        else:

            if(i == 0): # Take the first position estimation as the reference for the first fruit
                position_est1.append(position_est[i])
                continue
            else:
                coordinates = position_est[i]
                x_distance = np.abs(coordinates[0] - position_est[0][0])
                y_distance = np.abs(coordinates[1] - position_est[0][1])
                distance = np.sqrt(x_distance ** 2 + y_distance ** 2)
                if(distance < distance_threshold):
                    position_est1.append(coordinates)
                else:
                    position_est2.append(coordinates)

    # Merge position estimations
    position1 = merge_to_mean(position_est1, remove_outlier)
    position2 = merge_to_mean(position_est2, remove_outlier)

    # return the position estimations
    positions = []
    if(position1 is not None):
        positions.append(position1)
    if(position2 is not None):
        positions.append(position2)
    return positions
        

# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_map = target_pose_dict
    apple_est, lemon_est, pear_est, orange_est, strawberry_est = [], [], [], [], []
    target_est = {}
    
    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('pear'):
                pear_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('strawberry'):
                strawberry_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    remove_outlier = False
    use_Kmeans = False
    if len(apple_est) > 1:
        apple_est = sort_locations_and_merge(apple_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
    if len(lemon_est) > 1:
        lemon_est = sort_locations_and_merge(lemon_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
    if len(pear_est) > 1:
        pear_est = sort_locations_and_merge(pear_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
    if len(orange_est) > 1:
        orange_est = sort_locations_and_merge(orange_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)
    if len(strawberry_est) > 1:
        strawberry_est = sort_locations_and_merge(strawberry_est, distance_threshold = 0.3, remove_outlier = remove_outlier, use_Kmeans = use_Kmeans)

    for i in range(2):
        try:
            target_est['apple_'+str(i)] = {'y':apple_est[i][0], 'x':apple_est[i][1]}
        except:
            pass
        try:
            target_est['lemon_'+str(i)] = {'y':lemon_est[i][0], 'x':lemon_est[i][1]}
        except:
            pass
        try:
            target_est['pear_'+str(i)] = {'y':pear_est[i][0], 'x':pear_est[i][1]}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'y':orange_est[i][0], 'x':orange_est[i][1]}
        except:
            pass
        try:
            target_est['strawberry_'+str(i)] = {'y':strawberry_est[i][0], 'x':strawberry_est[i][1]}
        except:
            pass
    ########################################### 
    return target_est

if __name__ == "__main__":
    #Make sure the weight file corresponds to the test image.
    weight_path =  'network/scripts/model/best.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path, force_reload=True)    
    calibration_path = 'calibration/param/'
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    with open(base_dir/'fruit_estimates/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
    
    # estimate pose of targets in each detector output
    # {image_0:{apple:{y:YPOS,x:XPOS}}}
    target_map = {}        
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(base_dir, file_path, image_poses)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)
                     
    # save target pose estimations
    with open(base_dir/'fuit_estimates/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved')