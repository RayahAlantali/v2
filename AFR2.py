
# basic python packages
import numpy as np
import cv2
import os, sys
import time
import json
import copy

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components you developed in M2
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import CV components
sys.path.insert(0,"{}/network/".format(os.getcwd()))
sys.path.insert(0,"{}/network/scripts".format(os.getcwd()))

# import path planning components for M4
from path_planning.RRT import *

class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)

        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0],
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False,
                        'search': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.pred_count = 0
        self.notification = 'Press ENTER to start SLAM'
        #SEtting a timwe for 5 minutes
        self.count_down = 300
        self.start_time = time.time()
        self.clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.detector_output = np.zeros([240,320], dtype=np.uint8)
        if args.ckpt == "":
            self.detector = None
            self.network_vis = cv2.imread('pics/8bit/detector_splash.png')
        else:
            # self.detector = Detector(args.ckpt, use_gpu=False)
            self.network_vis = np.ones((240, 320,3))* 100
            self.grid = cv2.imread('grid.png')
        self.bg = pygame.image.load('pics/gui_mask.jpg')
        #Initialisng paramaters and arrays to be used
        self.robot_pose = np.array([0,0,0])
        self.paths = [[[0,0],[0.5,0.5]],[[0.5,0.5],[1,1]],[[1,1],[1,0.5]]]
        self.forward = False
        self.point_idx = 1
        self.waypoints = []
        self.wp = [0,0]
        self.min_dist = 50
        self.auto_path = False
        self.taglist = []
        #Defining parameters for SLAM
        self.P = np.zeros((3,3))
        self.marker_pos = np.zeros((2,10))
        self.lmc = 1e-6
        self.path_idx = 0

        #Contorl and travel parameters
        self.tick = 30
        self.turning_tick = 5
        self.boundary = 0.22

        #Add known markers and fruits from map to SLAM
        self.fruit_list, self.fruit_true_pos, self.aruco_true_pos = self.read_true_map(args.true_map)
        self.marker_pos = np.zeros((2,len(self.aruco_true_pos) + len(self.fruit_true_pos)))
        self.marker_pos, self.taglist, self.P = self.parse_slam_map(self.fruit_list, self.fruit_true_pos, self.aruco_true_pos)
        self.ekf.load_map(self.marker_pos, self.taglist, self.P)

        #Creating paths from the know search_liat
        self.search_list = self.read_search_list()
        print(f'Fruit search order: {self.search_list}')
        self.generate_paths()

    # wheel control
    def control(self):
        if args.play_data:
            lv, rv = self.pibot.set_velocity()
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'], self.tick, self.turning_tick)
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.clock = time.time()
        return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    def detect_fruit_pos(self, dictionary):
        measurements = []
        for fruit in dictionary.keys():
            x = dictionary[fruit]['x']
            y = dictionary[fruit]['y']
            lm_measurement = measure.Marker(np.array([x,y]),fruit)
            measurements.append(lm_measurement)
        return measurements

    def read_search_list(self):
        """Function reads the order of the target fruits returnf the target fruits in order
        """
        search_list = []
        with open('search_list.txt', 'r') as fd:
            fruits = fd.readlines()
            for fruit in fruits:
                search_list.append(fruit.strip())

        return search_list

    def generate_paths(self):
        #getting index of fruits to be searched
        fruit_list_dict = dict(zip(self.fruit_list,range(len(self.fruit_list))))
        all_fruits = [x for x in range(len(self.fruit_list))]
        search_fruits = [fruit_list_dict[x] for x in self.search_list]
        other_fruits = list((set(all_fruits) | set(search_fruits)) - (set(all_fruits) & set(search_fruits)))

        #Putting in the Markers
        obstacles = []
        for x,y in self.aruco_true_pos:
            obstacles.append([x + 1.5, y + 1.5])

        #Making the other fruits obstacles
        for idx in other_fruits:
            x,y = self.fruit_true_pos[idx]
            obstacles.append([x + 1.5, y + 1.5])

        all_obstacles = generate_path_obstacles(obstacles, self.boundary) #generating obstacles

        #starting robot pose and empty paths
        start = np.array([0,0]) + 1.5
        paths = []
        print(self.fruit_true_pos)
        for idx in search_fruits:
            location = copy.deepcopy(self.fruit_true_pos[idx])
            offset = 0.15
            #Stop in front of fruit
            if location[0] > 0 and location[1] > 0:
                location -= [offset, offset]
            elif location[0] > 0 and location[1] < 0:
                location -= [offset, -offset]
            elif location[0] < 0 and location[1] > 0:
                location -= [-offset, offset]
            else:
                location += [offset, offset]

            print(f' {self.fruit_list[idx]} at {location}')
            goal = np.array(location) + 1.5


            rrtc = RRT(start=start, goal=goal, width=3, height=3, obstacle_list=all_obstacles,
                    expand_dis=1, path_resolution=0.5)
            path = rrtc.planning()[::-1] #reverse path

            #printing path
            for i in range(len(path)):
                x, y = path[i]
                path[i] = [x - 1.5, y - 1.5]
            # print(f'The path is {path}')

            #adding paths
            paths.append(path)
            start = np.array(goal)
        self.paths = paths

    # SLAM with ARUCO markers
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)

        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # using computer vision to detect targets
    def detect_target(self):
        if self.command['inference'] and self.detector is not None:
            self.detector_output, self.network_vis, self.bounding_boxes, pred_count = self.detector.detect_single_image(self.img)
            self.command['inference'] = False
            self.file_output = (self.detector_output, self.ekf)
            self.notification = f'{pred_count} fruits detected'

    # save raw images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
            scale /= 2
        fileB = "{}baseline.txt".format(datadir)
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # function to save bounding box info
    def bounding_box_output(self, box_list):
        import json
        with open(f'lab_output/pred_{self.pred_count}.txt', "w") as f:
            json.dump(box_list, f)
            self.pred_count += 1

    # save the SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False
        # save inference with the matching robot pose and detector labels
        if self.command['save_inference']:
            if self.file_output is not None:
                image = cv2.cvtColor(self.file_output[0], cv2.COLOR_BGR2RGB)
                self.pred_fname = self.output.write_image(image,
                                                        self.file_output[1])
                self.bounding_box_output(self.bounding_boxes) #save bounding box text file
                self.notification = f'Prediction is saved to pred_{self.pred_count-1}.png'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_inference'] = False

    def parse_slam_map(self, fruit_list, fruits_true_pos, aruco_true_pos):

        #Aruco Markers
        for i, pos in enumerate(aruco_true_pos):
            self.taglist.append(i + 1)
            self.marker_pos[0][i] = pos[0]
            self.marker_pos[1][i] = pos[1]
            #Adding to the covariance matrix
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
            self.P[-2,-2] = self.lmc**2
            self.P[-1,-1] = self.lmc**2

        #Known Fruits
        for i,pos in enumerate(fruits_true_pos):
            self.taglist.append(fruit_list[i]) #adding tag
            self.marker_pos[0][i + 10] = pos[0]
            self.marker_pos[1][i + 10] = pos[1]
            #Adding it to the covariance matrix
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
            self.P[-2,-2] = self.lmc**2
            self.P[-1,-1] = self.lmc**2
        return self.marker_pos, self.taglist, self.P

    def read_true_map(self,fname):
        """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

        @param fname: filename of the map
        @return:
            1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
            2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
            3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
        """
        with open(fname, 'r') as fd:
            gt_dict = json.load(fd)
            fruit_list = []
            fruit_true_pos = []
            aruco_true_pos = np.empty([10, 2])

            # remove unique id of targets of the same type
            for key in gt_dict:
                x = np.round(gt_dict[key]['x'], 1)
                y = np.round(gt_dict[key]['y'], 1)

                if key.startswith('aruco'):
                    if key.startswith('aruco10'):
                        aruco_true_pos[9][0] = x
                        aruco_true_pos[9][1] = y
                    else:
                        marker_id = int(key[5])
                        aruco_true_pos[marker_id-1][0] = x
                        aruco_true_pos[marker_id-1][1] = y
                else:
                    fruit_list.append(key[:-2])
                    if len(fruit_true_pos) == 0:
                        fruit_true_pos = np.array([[x, y]])
                    else:
                        fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

            return fruit_list, fruit_true_pos, aruco_true_pos
    # paint the GUI
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20
 
        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(240, 240+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (h_pad,2*v_pad+240))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view,
                                position=(h_pad, v_pad))

        # Using a grid image as the interface for waypoints
        gui_grid = cv2.resize(self.grid,
                                   (320, 480), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, gui_grid,
                                position=(2*h_pad+320, v_pad))
        #Defining colours to use for the GUI
        black = pygame.Color(0,0,0)
        red = pygame.Color(255,0,0)
        blue = pygame.Color(0,0,255)
        green = pygame.Color(102,204,0)
        yellow = pygame.Color(255,255,0)
        orange = pygame.Color(255,165,0)
        magenta = pygame.Color(255,0,255)
        grey = pygame.Color(220,220,220)
        purple = pygame.Color(128,0,128)

        #Painting Marker positions on the grid
        for marker in self.aruco_true_pos:
            x = int(160+marker[0]*80 )
            y = int(160 - marker[1]*80)
            pygame.draw.circle(canvas, purple, (2*h_pad+320 + x,v_pad + y),self.boundary*80,0)
            pygame.draw.rect(canvas, black, (2*h_pad+320 + x - 5,v_pad + y - 5,10,10))
        #Painting the fruits on the grid
        for i, fruit in enumerate(self.fruit_list):
            if fruit == 'apple':
                colour = red
            elif fruit == 'lemon':
                colour = yellow
            elif fruit == 'orange':
                colour = orange
            elif fruit == 'pear':
                colour = green
            elif fruit == 'strawberry':
                colour = magenta

            x = int(160+self.fruit_true_pos[i][0]*80)
            y = int(240 - self.fruit_true_pos[i][1]*80)
            #Drawing the current fruit in the list
            pygame.draw.circle(canvas, colour, (2*h_pad+320 + x,v_pad + y),4)
            if fruit not in self.search_list:
                pygame.draw.circle(canvas, grey, (2*h_pad+320 + x,v_pad + y),self.boundary*80)
            else:
                pygame.draw.circle(canvas, black, (2*h_pad+320 + x,v_pad + y),0.5*80, 2)
            
        #Painting the robot on the grid
        x = int(self.robot_pose[0]*80 + 120)
        y = int(120 - self.robot_pose[1]*80)
        x2 = int(x + 20*np.cos(self.robot_pose[2]))
        y2 = int(y - 20*np.sin(self.robot_pose[2]))
        pygame.draw.rect(canvas, blue, (2*h_pad+320 + x - 5,v_pad + y - 5,10,10))
        pygame.draw.line(canvas, black, (2*h_pad+320 + x,240 + 2*v_pad + y),(2*h_pad +320+ x2,v_pad + y2))

        #Draw the waypoint
        x = int(self.wp[0]*80 + 120)
        y = int(120 - self.wp[1]*80)
        pygame.draw.line(canvas, red,(2*h_pad+320 + x-5,v_pad + y-5), (2*h_pad+320 + x + 5,v_pad + y + 5))
        pygame.draw.line(canvas, red,(2*h_pad+320 + x+ 5,v_pad + y-5), (2*h_pad+320 + x - 5,v_pad + y + 5))

        #Draw path
        for path in self.paths:
            for i in range(len(path)-1):
                x = int(path[i][0]*80 + 120)
                y = int(120 - path[i][1]*80)
                x2 = int(path[i+1][0]*80 + 120)
                y2 = int(120 - path[i+1][1]*80)
                pygame.draw.line(canvas, blue, (2*h_pad+320 + x,v_pad + y),(2*h_pad+320 + x2,v_pad + y2))


        self.put_caption(canvas, caption='Grid Map',
                         position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='SLAM', position=(h_pad, 2*v_pad))
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))
        self.put_caption(canvas, caption='Detector', position=(3*h_pad + 2*320,v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Up !!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)

    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'][0] = min(self.command['motion'][0]+1, 1)
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'][0] = max(self.command['motion'][0]-1, -1)
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'][1] = min(self.command['motion'][1]+1, 1)
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'][1] = max(self.command['motion'][1]-1, -1)
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # AFR
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                self.path_idx = 0
                self.point_idx = 1
                self.waypoints = self.paths[self.path_idx] #set first path
                self.wp = self.waypoints[self.point_idx] #set waypoint to second point in path
                self.auto_path = True
                print(f"Moving to new waypoint {self.wp}")
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()

    def drive_robot(self):
        waypoint_x = self.wp[0]
        waypoint_y = self.wp[1]
        #Updating the robots psoe
        self.robot_pose = self.ekf.get_state_vector()
        robot_x = self.robot_pose[0]
        robot_y = self.robot_pose[1]

        self.distance = np.sqrt((waypoint_x-robot_x)**2 + (waypoint_y-robot_y)**2) #calculates distance between robot and waypoint

        robot_theta = self.robot_pose[2]
        waypoint_angle = np.arctan2((waypoint_y-robot_y),(waypoint_x-robot_x))
        theta1 = robot_theta - waypoint_angle
        if waypoint_angle < 0:
            theta2 = robot_theta - waypoint_angle - 2*np.pi
        else:
            theta2 = robot_theta - waypoint_angle + 2*np.pi

        if abs(theta1) > abs(theta2):
            self.theta_error = theta2
        else:
            self.theta_error = theta1

        if self.forward == False:
            #Update turning tick speed depending on theta_error to waypoint
            self.turning_tick = int(abs(5 * self.theta_error) + 3)
            if self.theta_error > 0:
                self.command['motion'] = [0,-1]
                self.notification = 'Robot is turning right'

            if self.theta_error < 0:
                self.command['motion'] = [0,1]
                self.notification = 'Robot is turning left'

        # stop turning if less than threshold
        if not self.forward:
            if abs(self.theta_error)  < 0.05:
                self.command['motion'] = [0,0]
                self.notification = 'Robot stopped turning'
                self.forward = True #go forward now
                return

        #Driving forward
        if self.forward:
            #Update tick speed depending on distance to waypoint
            self.tick = int(10 * self.distance  + 30)

            #Checking if distance is increasing, stop driving
            if self.distance > self.min_dist + 0.1:
                self.command['motion'] = [0,0]
                self.notification = 'Robot stopped moving'
                self.forward = False
                self.min_dist = 50
                return

            # Distance is decreasing
            else:
                #Drive until goal arrived
                distance_threshold = 0.1 #0.05
                if self.distance < distance_threshold:
                    self.command['motion'] = [0,0]
                    self.notification = 'Robot arrived'
                    self.forward = False
                    self.min_dist = 50

                    #Check if last path and last waypoint reached
                    if self.point_idx == len(self.waypoints) - 1: #reached last wp of path
                        if self.path_idx == len(self.paths) - 1: #stop pathing
                            self.auto_path = False
                        else: #Increment path and reset idx
                            self.path_idx += 1
                            self.waypoints = self.paths[self.path_idx]
                            self.point_idx = 1 
                            self.wp = self.waypoints[self.point_idx]
                        self.pibot.set_velocity([0,0],time = 3)
                    else:
                        self.point_idx += 1
                        self.wp = self.waypoints[self.point_idx]
                    print(f"Moving to new waypoint {self.wp}")
                    return

                else:
                    #ReAdjust angle if theta_error increased
                    if abs(self.theta_error) > 15/57.3 and self.distance > 0.15: #0.2
                        self.command['motion'] = [0,0]
                        self.notification = 'Readjusting angle'
                        self.forward = False
                        self.min_dist = 50
                        return

                    self.min_dist = self.distance
                    self.command['motion'] = [1,0]
                    self.notification = 'Robot moving forward'


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    parser.add_argument("--true_map", default="M4_true_map_3fruits.txt")
    parser.add_argument("--ckpt", default='yolo-sim.pt')
    args, _ = parser.parse_known_args()

    pygame.font.init()
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)

    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png').convert())
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png').convert()
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png').convert(),
                     pygame.image.load('pics/8bit/pibot2.png').convert(),
                     pygame.image.load('pics/8bit/pibot3.png').convert(),
                    pygame.image.load('pics/8bit/pibot4.png').convert(),
                     pygame.image.load('pics/8bit/pibot5.png').convert()]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    operate.notification = 'SLAM is running'
    operate.ekf_on = True
    while start:
        operate.update_keyboard()
        operate.take_pic()
        if operate.auto_path:
            operate.drive_robot()
        drive_meas = operate.control()

        operate.update_slam(drive_meas)
        operate.robot_pose = operate.ekf.robot.state
        operate.record_data()
        operate.save_image()
        operate.detect_target()
        # visualise
        operate.draw(canvas)
        pygame.display.update()




