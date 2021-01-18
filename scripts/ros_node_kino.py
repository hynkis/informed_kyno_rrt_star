#!/usr/bin/env python3
"""
ROS node for the Informed RRT* path planning

author: Hyunki Seong

Reference: Informed RRT*: Optimal Sampling-based Path planning Focused via
Direct Sampling of an Admissible Ellipsoidal Heuristic
https://arxiv.org/pdf/1404.2334.pdf

@ Map size
    - x : [-0.3 ~ 9.0]
    - y : [-2.0 ~ 7.0]
@ Start point
    - [0, 0]
@ Goal point
    - [2.34, 3.11]

"""

import time
import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import parmap

import rospy
import rospkg

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from scipy.spatial.transform import Rotation

from plot_utils import Plot_utils

try:
    import reeds_shepp_path_planning
except ImportError:
    raise

rospack = rospkg.RosPack()

TRAJECTORY_DATA_PATH = rospack.get_path('informed_kyno_rrt_star')+'/data/world_3'
SAVE_DATA = False
DATA_I = 0

# ============================ #
# ----- Global functions ----- #
# ============================ #

def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def calc_distance_and_angle(fromNode, toNode):
    dx = toNode.x - fromNode.x
    dy = toNode.y - fromNode.y
    d = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    return d, theta

def euler_to_quaternion(rot_x, rot_y, rot_z):
    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=False)

    # Convert to quaternions
    rot_quat = rot.as_quat() #[x,y,z,w]
    
    return rot_quat

def quaternion_to_euler(q):
    """
    q = [x,y,z,w] to e = [zxy]
    """
    # Create a rotation object from Euler angles specifying axes of rotation
    rot = Rotation.from_quat(q)
    # Convert to euler
    rot_euler = rot.as_euler('zxy', degrees=False)
    
    return rot_euler

class Node:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.path_x = []
        self.path_y = []
        self.path_yaw = []
        self.cost = 0.0
        self.parent = None # Node, not index

class ReedsSheppRRTStar(Plot_utils):
    def __init__(self, start, goal, randArea, eta, max_rewire_num, car_radius, curvature_limit, reeds_shepp_step_size):
        """
            - start    : [start x, y]
            - goal     : [goal x, y]
            - randArea : [min_pose, max_pose]
        """
        # ROS init
        rospy.init_node('informed_rrt_star')
        self.rate = rospy.Rate(30)

        # Parameters
        self.map = None
        self.obstacle_list = []
        self.node_list = None

        self.startNode = Node(start[0], start[1], start[2])
        self.goalNode  = Node(goal[0], goal[1], goal[2])

        self.max_iter = 500
        self.max_rewire_num = max_rewire_num
        self.goal_sample_rate = 0 # % of sampling goal point directly

        self.min_rand = randArea[0] # min random value
        self.max_rand = randArea[1] # max random value

        # RRT*
        self.near_radius = 50.0
        self.eta = eta  # expand distance
        # Reeds Shepp
        self.curvature_limit       = curvature_limit
        self.reeds_shepp_step_size = reeds_shepp_step_size

        self.goal_yaw_thres = np.deg2rad(1.0)
        self.goal_xy_thres  = 0.5

        # Subscriber
        self.sub_map = rospy.Subscriber("map", OccupancyGrid, self.callback_map)
        self.sub_init_pose = rospy.Subscriber("initialpose", PoseWithCovarianceStamped, self.callback_init_pose)
        self.sub_goal_pose = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.callback_goal_pose)
        # Publisher
        self.pub_path  = rospy.Publisher('/informed_kyno_rrt_star/informed_rrt_star', Path, queue_size=10)
        self.pub_local_obs = rospy.Publisher('/visual/obstacle', Marker, queue_size=10)
        self.pub_current_pose = rospy.Publisher('/visual/current_pose', Marker, queue_size=1)
        self.pub_goal_pose    = rospy.Publisher('/visual/goal_pose', Marker, queue_size=1)
        self.pub_node  = rospy.Publisher('/visual/node', Marker, queue_size=10)
        
        self.pub_tree = rospy.Publisher('/visual/tree', MarkerArray, queue_size=10)
        # self.pub_reeds = rospy.Publisher('/visual/reeds', Marker, queue_size=10)
        # self.pub_final_reeds = rospy.Publisher('/visual/final_reeds', Marker, queue_size=10)
        self.pub_reeds = rospy.Publisher('/visual/reeds', PoseArray, queue_size=10)
        self.pub_final_reeds = rospy.Publisher('/visual/final_reeds', PoseArray, queue_size=10)
        self.pub_ellipse = rospy.Publisher('/visual/ellipse', Marker, queue_size = 1)

        # Map
        # - Initialize map
        while self.map is None:
            # Wait until map is initialized
            print("wait map")
            continue
        self.grid           = np.reshape(np.array(self.map.data), (self.map.info.width, self.map.info.height))
        self.map_resolution = self.map.info.resolution
        self.origin         = self.map.info.origin
        print("Origin :", self.origin)

        # Car
        self.car_radius = car_radius
        self.car_volume_X, self.car_volume_Y = np.ogrid[-round(self.car_radius/self.map_resolution):round(self.car_radius/self.map_resolution)+1,
                                                        -round(self.car_radius/self.map_resolution):round(self.car_radius/self.map_resolution)+1 ]


    # ===================== #
    # ----- Callbacks ----- #
    # ===================== #
    
    def callback_map(self, msg_map):
        self.map = msg_map
        print("subscribe map")
    
    def callback_init_pose(self, msg_pose):
        self.startNode.x = msg_pose.pose.pose.position.x
        self.startNode.y = msg_pose.pose.pose.position.y

    def callback_goal_pose(self, msg_pose):
        q = [msg_pose.pose.orientation.x, msg_pose.pose.orientation.y, msg_pose.pose.orientation.z, msg_pose.pose.orientation.w]
        rot_z, rot_x, rot_y = quaternion_to_euler(q)
        
        self.goalNode.x = msg_pose.pose.position.x
        self.goalNode.y = msg_pose.pose.position.y
        self.goalNode.yaw = rot_z
        print("Received Goal pose :", self.goalNode.x, self.goalNode.y, np.rad2deg(self.goalNode.yaw))
        # Sampling
        found_path = self.search()
        
    # =============== #
    # ----- Map ----- #
    # =============== #

    def get_map_index(self, x, y):
        """
        Transform from x, y position in /map frame to numpy grid array x_index, y_index
        """ 
        x_ind = int((x-self.origin.position.x) // self.map_resolution)
        y_ind = int((y-self.origin.position.y) // self.map_resolution)
        # x, y are flipped in map_value
        return y_ind, x_ind

    def get_map_pose(self, ix, iy):
        """
        Transform from ix, iy to x, y
        """
        # x, y are flipped in map_value
        x = (iy+1) * self.map_resolution + self.origin.position.x
        y = (ix+1) * self.map_resolution + self.origin.position.y
        return x, y


    # ===================== #
    # ----- Functions ----- #
    # ===================== #

    def search(self):
        # for checking inference time
        np_time_tic = time.time()

        self.node_list = [self.startNode]
        cBest       = float('inf') # max length starts from inf
        rewire_cnt  = 0
        solutionSet = set()
        path        = None

        # for saving data
        global DATA_I

        # Compute the sampling space (Ellipsoid region)
        # - min cost == Min distance from start to goal
        cMin = calc_dist(self.startNode.x, self.startNode.y, self.goalNode.x, self.goalNode.y)
        # - center btw start and goal
        xCenter = np.array([[(self.startNode.x + self.goalNode.x) / 2.0],
                            [(self.startNode.y + self.goalNode.y) / 2.0], [0]])
        a1 = np.array([[(self.goalNode.x - self.startNode.x) / cMin],
                       [(self.goalNode.y - self.startNode.y) / cMin], [0]])
        e_theta = math.atan2(a1[1], a1[0])

        # TODO: Analyze below
        # first column of identity matrix transposed
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        # M = a1 @ id1_t
        M = np.matmul(a1, id1_t)
        U, S, Vh = np.linalg.svd(M, True, True)
        C = np.dot(np.dot(U, np.diag(
            [1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])),
                   Vh)

        for i in range(self.max_iter):
            tic_for = time.time()

            # ----- Algorithm ----- #
            # Sampling a node (rnd : [x, y])
            rnd         = self.informed_sample(cBest, cMin, xCenter, C)
            # Find the nearest node in the node list
            nearest_node_ind = self.get_nearest_node_ind(rnd)
            nearestNode = self.node_list[nearest_node_ind]
            # Steer
            newNode = self.steer(nearestNode, rnd)
            if newNode is None:
                print("new Node is None")
                continue
            # Stretch
            if self.check_collision_free(newNode):
                # Find nearest node in tree
                # tic = time.time()

                near_inds = self.find_near_nodes(newNode)
                newNode   = self.choose_parent(newNode, near_inds)
                # print("choose parent time :", time.time() - tic)

                self.node_list.append(newNode)

                # Rewiring
                self.rewire(newNode, near_inds)

                if cBest < float('inf'):
                    rewire_cnt += 1

                # Try direct goal path
                self.try_goal_path(newNode)

                # Goal check
                if newNode:
                    lastNode = self.search_best_goal_node()
                    if lastNode:
                        # for checking inference time
                        inference_time = time.time() - np_time_tic
                        print("Found lastNode :", lastNode.x, lastNode.y, np.rad2deg(lastNode.yaw))
                        final_path = self.get_final_path(lastNode)
                        path_cost = final_path[0].cost
                        path_time = inference_time
                        cBest = path_cost

                        if len(final_path) != 0:
                            # Save Path_x, y, yaw / Cost / Time as npz
                            # - path
                            np_data = []
                            for node in reversed(final_path):
                                for x, y, yaw in zip(node.path_x, node.path_y, node.path_yaw):
                                    np_data_array = np.array([x, y, yaw, path_cost, path_time])
                                    if len(np_data) == 0:
                                        np_data = np_data_array
                                    else:
                                        np_data = np.vstack([np_data, np_data_array])

                            # - save as npz
                            if SAVE_DATA:
                                trajectory_data_path = TRAJECTORY_DATA_PATH + '_' + str(DATA_I) + '.csv'
                                dataframe = pd.DataFrame(np_data)
                                dataframe.to_csv(trajectory_data_path, header=False, index=False)
                                print("Save done! cost, time :", path_cost, path_time)
                                DATA_I += 1

                        # self.publish_final_reeds_shepp_path(final_path)
                        self.publish_final_reeds_shepp_path_posearray(final_path)
                        # return final_path

                        self.publish_ellipse(xCenter, cBest, cMin, e_theta)

            # Visualization
            # - tree
            self.publish_point([(self.startNode.x, self.startNode.y)], publisher=self.pub_current_pose, color=(255,0,0))
            self.publish_point([(self.goalNode.x, self.goalNode.y)], publisher=self.pub_goal_pose, color=(0,0,255))
            self.publish_tree(self.node_list)
            # self.publish_reeds_shepp_path(self.node_list)
            self.publish_reeds_shepp_path_posearray(self.node_list)

            # print("For loop time :", time.time() - tic_for)
            if rewire_cnt > self.max_rewire_num:
                print("DONE!!!")
                time.sleep(3)
                break


        return path

    def steer(self, fromNode, toNode, update_parent=True):
        # Calculate distance and theta for stretching
        # dist  = calc_dist(fromNode.x, fromNode.y, toNode.x, toNode.y)
        # theta = math.atan2(toNode.y - fromNode.y, toNode.x - fromNode.x)

        px, py, pyaw, mode, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            fromNode.x, fromNode.y, fromNode.yaw,
            toNode.x, toNode.y, toNode.yaw, self.curvature_limit,
            step_size=self.reeds_shepp_step_size
        )

        if not px:
            return None

        # Generate new Node from Reeds Shepp Planning result
        newNode = copy.deepcopy(fromNode)
        newNode.x   = px[-1]
        newNode.y   = py[-1]
        newNode.yaw = pyaw[-1]

        newNode.path_x = px
        newNode.path_y = py
        newNode.path_yaw = pyaw
        newNode.cost += sum([abs(l) for l in course_lengths])
        # TODO: Add backward driving penalty

        # Update parent except during rewiring
        if update_parent:
            newNode.parent = fromNode

        return newNode

    def rewire(self, newNode, near_inds):
        """
        Calculate cost of near nodes and rewire to less-cost node
        """
        n_node = len(self.node_list)
        for i in near_inds:
            nearNode = self.node_list[i]
            edgeNode = self.steer(newNode, nearNode, update_parent=True) # from new to near node in rewiring
            if not edgeNode:
                continue
            # edgeNode.cost = self.calc_new_cost(newNode, nearNode)

            if edgeNode.cost < nearNode.cost:
                # Collision check from newNode to nearNode (Rewiring)
                if self.check_collision_free(edgeNode):
                    nearNode.parent = edgeNode.parent # nearNode is updated by edgeNode.
                    nearNode.path_x = edgeNode.path_x
                    nearNode.path_y = edgeNode.path_y
                    nearNode.path_yaw = edgeNode.path_yaw
                    nearNode.cost = edgeNode.cost

    def calc_new_cost(self, fromNode, toNode):
        _, _, _, _, course_lengths = reeds_shepp_path_planning.reeds_shepp_path_planning(
            fromNode.x, fromNode.y, fromNode.yaw,
            toNode.x, toNode.y, toNode.yaw, self.curvature_limit,
            step_size=self.reeds_shepp_step_size
        )
        if not course_lengths:
            return float("inf")
        
        return fromNode.cost + sum([abs(l) for l in course_lengths])


    # ==================== #
    # ----- Sampling ----- #
    # ==================== #

    def informed_sample(self, cMax, cMin, xCenter, C):
        # If a path to goal was found, sampling in an ellipsoid region
        # Prevent math-error when they are ~equal
        if cMax < cMin:
            cMax = cMin

        if cMax < float('inf'):
            r = [cMax*0.5,
                 math.sqrt(cMax**2 - cMin**2)*0.5,
                 math.sqrt(cMax**2 - cMin**2)*0.5]
            L = np.diag(r)
            xBall = self.sample_unit_ball()
            # Informed sampled x, y
            rnd = np.dot(np.dot(C, L), xBall) + xCenter # ellipsoid
            # Random yaw
            rnd_yaw = random.uniform(-math.pi, math.pi)
            # rnd = [rnd[(0,0)], rnd[(1,0)]]
            rnd = Node(rnd[(0,0)], rnd[(1,0)], rnd_yaw) # Node, not list

        # If no path was found
        else:
            rnd = self.sample_free_space()

        return rnd

    def sample_unit_ball(self):
        """
        Sampling in ellipsoid space
        """
        a = random.random()
        b = random.random() 

        if b < a: a, b = b, a # should be a < b

        sample = (b * math.cos(2 * math.pi * a/b),
                  b * math.sin(2 * math.pi * a/b))
        
        return np.array([[sample[0]], [sample[1]], [0]])

    def sample_free_space(self):
        """
        Sampling in random space
        """
        # Sampling nodes (10% at the Goal point / 90% at the Random point)
        if random.randint(0, 100) > self.goal_sample_rate:
            # rnd = [random.uniform(self.min_rand, self.max_rand),
            #        random.uniform(self.min_rand, self.max_rand)]
            rnd = Node(random.uniform(self.min_rand, self.max_rand),
                       random.uniform(self.min_rand, self.max_rand),
                       random.uniform(-math.pi, math.pi)
                       )
        else:
            rnd = Node(self.goalNode.x, self.goalNode.y, self.goalNode.yaw)

        return rnd


    # ================ #
    # ----- Node ----- #
    # ================ #

    def find_near_nodes(self, newNode):
        # Refer RRT*
        n_node = len(self.node_list)
        if n_node > 1:
            radius = self.near_radius * math.sqrt((math.log(n_node) / n_node))
            d_list = [calc_dist(node.x, node.y, newNode.x, newNode.y) for node in self.node_list]
            near_inds = [d_list.index(i) for i in d_list if i <= radius]
        else:
            near_inds = [0]
        return near_inds

    def get_nearest_node_ind(self, rnd):
        dist_list = [calc_dist(node.x, node.y, rnd.x, rnd.y) for node in self.node_list]
        min_ind = dist_list.index(min(dist_list))

        return min_ind

    def choose_parent(self, newNode, near_inds):
        """
        Choose parent node
            - find nearest node inds
            - calculate cost
            - find minimum-cost node as parent
        """
        tic = time.time()

        num_near_inds = len(near_inds)
        if num_near_inds == 0:
            print("no near nodes")
            return newNode
        
        costs = []
        tNode_list = []
        for i in near_inds:
            nearNode = self.node_list[i]
            tNode = self.steer(nearNode, newNode)
            tNode_list.append(tNode)
            if tNode and self.check_collision_free(tNode):
                costs.append(self.calc_new_cost(nearNode, newNode))
            else:
                costs.append(float('inf'))

        min_cost = min(costs)
        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return newNode

        min_ind  = near_inds[costs.index(min_cost)]
        min_cost_tNode = tNode_list[min_ind]
        # Wiring min cost node with newNode as parent 
        newNode = self.steer(self.node_list[min_ind], newNode) # updated parent in steer.
        # newNode.parent = self.node_list[min_ind]
        newNode.cost = min_cost

        # print("choose parent time :", time.time() - tic)

        return newNode

    def calc_collision_free_dist(self, newNode, near_inds):
        dist_list = []
        print("near_inds :", near_inds)
        for i in near_inds:
            dist = calc_dist(self.node_list[i].x, self.node_list[i].y, newNode.x, newNode.y)
            collision_free = self.check_collision_free(self.node_list[i], newNode)
            if collision_free:
                dist_list.append(self.node_list[i].cost + dist)
            else:
                dist_list.append(float('inf'))
        return dist_list

    def search_best_goal_node(self):
        """
        """
        goal_inds = []
        for (i, node) in enumerate(self.node_list):
            if calc_dist(node.x, node.y, self.goalNode.x, self.goalNode.y) <= self.goal_xy_thres:
                goal_inds.append(i)

        # Heading check
        final_goal_inds = []
        for i in goal_inds:
            if abs(self.node_list[i].yaw - self.goalNode.yaw) <= self.goal_yaw_thres:
                final_goal_inds.append(i)

        if not final_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_inds])
        print("Min cost :", min_cost)
        for i in final_goal_inds:
            if self.node_list[i].cost == min_cost:
                return self.node_list[i]

        return None

    def try_goal_path(self, node):
        """
        Try current node to Goal
        """
        newNode = self.steer(node, self.goalNode)
        # If no direct path to goal
        if newNode is None:
            return
        # If there is a direct path to goal
        if self.check_collision_free(newNode):
            self.node_list.append(newNode)


    # ================ #
    # ----- Path ----- #
    # ================ #
    def get_final_path(self, node):
        """
        Backpropagate solution path
        Path is the list of Node
        """
        # path = [self.goalNode]
        path = []
        point_xy = []
        while node.parent is not None:
            path.append(node)
            point_xy.append(("pt", (node.x, node.y), "last_point", (node.path_x[-1], node.path_y[-1]), "init_point", (node.path_x[0], node.path_y[0])))
            node = node.parent
        path.append(self.startNode)
        point_xy.append((self.startNode.x, self.startNode.y))

        return path
        
    # def get_final_path(self, last_ind):
    #     """
    #     Backpropagate solution path
    #     Path is the list of Node
    #     """
    #     path = [self.goalNode]
    #     while self.node_list[last_ind].parent is not None:
    #         node = self.node_list[last_ind]
    #         path.append(node)
    #         last_ind = node.parent
    #     path.append(self.startNode)

    #     return path

    def get_path_len(self, path):
        path_len = 0
        num_node = len(path)
        for i in range(1, num_node):
            path_len += calc_dist(path[i].x, path[i].y, path[i-1].x, path[i-1].y)

        return path_len


    # ===================== #
    # ----- Collision ----- #
    # ===================== #

    def check_collision_free(self, toNode):
        """
        Check collision free of path_x and path_y in 'toNode'
        """
        for center_x, center_y in zip(toNode.path_x, toNode.path_y):
            # Get grid value
            # - centerline
            center_ix, center_iy = self.get_map_index(center_x, center_y)
            res = self.grid[center_ix + self.car_volume_X, center_iy + self.car_volume_Y] # 0: free | -1, 100: collision
            if res.any() == True:
                # print("Collision :", ix, iy, self.grid[ix, iy])
                return False # collision

        return True
    
    def steer_eta(self, nearNode, randNode, eta):
        # Calculate distance and theta for stretching
        dist  = calc_dist(nearNode.x, nearNode.y, randNode.x, randNode.y)
        theta = math.atan2(randNode.y - nearNode.y, randNode.x - nearNode.x)

        if dist < eta:
            nextNode = copy.deepcopy(randNode)
        else:
            nextNode = copy.deepcopy(nearNode)
            nextNode.x += math.cos(theta) * eta
            nextNode.y += math.sin(theta) * eta
        
        return nextNode

    # ======================= #
    # ----- Visualizer ----- #
    # ======================= #
    def publish_point(self, xy_list, publisher, color=(176, 60, 193)):

        # Publish wpt as MarkerArray
        msg_point = Marker()
        msg_point.header.frame_id= "/map"
        msg_point.header.stamp= rospy.Time.now()
        msg_point.ns= "spheres"
        msg_point.action= Marker.ADD
        msg_point.pose.orientation.w= 1.0

        msg_point.id = 0
        msg_point.type = Marker.SPHERE_LIST


        # POINTS markers use x and y scale for width/height respectively
        msg_point.scale.x = 0.1
        msg_point.scale.y = 0.1
        msg_point.scale.z = 0.1

        # Points are green
        msg_point.color.a = 1.0 # Don't forget to set the alpha!
        msg_point.color.r = color[0]/255.
        msg_point.color.g = color[1]/255.
        msg_point.color.b = color[2]/255.

        msg_point_list = []

        for x, y in xy_list:
            msg_p = Point()
            msg_p.x = x
            msg_p.y = y
            msg_point_list.append(msg_p)

        msg_point.points = msg_point_list
        publisher.publish(msg_point)

    def publish_tree(self, node_list):
        # Tree
        msg_tree = MarkerArray()
        # Point
        msg_point = Marker()
        msg_point.header.frame_id= "/map"
        msg_point.header.stamp= rospy.Time.now()
        msg_point.ns= "spheres"
        msg_point.action= Marker.ADD
        msg_point.pose.orientation.w= 1.0

        msg_point.id = 0
        msg_point.type = Marker.SPHERE_LIST

        # POINTS markers use x and y scale for width/height respectively
        msg_point.scale.x = 0.1
        msg_point.scale.y = 0.1
        msg_point.scale.z = 0.1

        # Points are green
        msg_point.color.a = 0.5 # Don't forget to set the alpha!
        msg_point.color.r = 0/255.
        msg_point.color.g = 0/255.
        msg_point.color.b = 255/255.

        id = 1
        for node in node_list:
            # Point
            msg_p = Point()
            msg_p.x = node.x
            msg_p.y = node.y
            msg_point.points.append(msg_p)
            
            # Tree
            if node.parent is not None:
                # Edge between nodes
                path = Marker()
                path.header.frame_id = "map"
                path.header.stamp = rospy.get_rostime()
                path.ns = "markers"
                path.id = id
                id += 1
                path.type = path.LINE_STRIP
                path.action = path.ADD
                path.scale.x = 0.03
                
                path.color.a = 1.0
                path.color.r = 255/255.
                path.color.g = 127/255.
                path.color.b = 0/255.

                path.lifetime = rospy.Duration()
                path.pose.orientation.w = 1.0

                # parentNode = self.node_list[node.parent]
                parentNode = node.parent
                p1 = Point()
                p1.x = parentNode.x
                p1.y = parentNode.y
                p1.z = 0.02
                path.points.append(p1)

                p2 = Point()
                p2.x = node.x
                p2.y = node.y
                p2.z = 0.02
                path.points.append(p2)

                msg_tree.markers.append(path)

        # self.pub_tree.publish(msg_tree)
        self.pub_node.publish(msg_point)

    def publish_ellipse(self, xCenter, cBest, cMin, etheta):
    
        # Prevent math-error when they are ~equal
        if cBest < cMin:
            cBest = cMin

        a = math.sqrt(cBest ** 2 - cMin ** 2) / 2.0
        b = cBest / 2.0
        angle = math.pi / 2.0 - etheta
        cx = xCenter[0]
        cy = xCenter[1]
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        R = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
        # fx = R @ np.array([x, y])
        fx = np.matmul(R, np.array([x, y]))
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()

        ellipse = Marker()
        ellipse.header.frame_id = "map"
        ellipse.header.stamp = rospy.get_rostime()
        ellipse.ns = "markers"
        ellipse.id = 1
        ellipse.type = ellipse.LINE_STRIP
        ellipse.action = ellipse.ADD

        ellipse.scale.x = 0.03 
        #ellipse.scale.y = 0.03
        ellipse.color.r = 0.0
        ellipse.color.g = 0.3
        ellipse.color.b = 7.0
        ellipse.color.a = 1.0
        ellipse.lifetime = rospy.Duration()
        ellipse.pose.orientation.w = 1.0

        for (x, y) in zip(px, py):
            p1 = Point()
            p1.x = x
            p1.y = y
            p1.z = 0
            ellipse.points.append(p1)
        
        self.pub_ellipse.publish(ellipse)

    def publish_reeds_shepp_path(self, node_list):
        # Publish wpt as MarkerArray
        msg_reeds = Marker()
        msg_reeds.header.frame_id= "/map"
        msg_reeds.header.stamp= rospy.Time.now()
        msg_reeds.ns= "spheres"
        msg_reeds.action= Marker.ADD
        msg_reeds.pose.orientation.w= 1.0

        msg_reeds.id = 0
        msg_reeds.type = Marker.SPHERE_LIST


        # POINTS markers use x and y scale for width/height respectively
        msg_reeds.scale.x = 0.1
        msg_reeds.scale.y = 0.1
        msg_reeds.scale.z = 0.1

        # Points are green
        msg_reeds.color.a = 1.0 # Don't forget to set the alpha!
        msg_reeds.color.r = 0/255.
        msg_reeds.color.g = 255/255.
        msg_reeds.color.b = 0/255.

        for node in node_list:
            # Tree
            if node.parent is not None:
                for x, y in zip(node.path_x, node.path_y):
                    msg_p = Point()
                    msg_p.x = x
                    msg_p.y = y
                    msg_reeds.points.append(msg_p)

        self.pub_reeds.publish(msg_reeds)

    def publish_reeds_shepp_path_posearray(self, node_list):
        # Publish wpt as PoseArray
        msg_reeds = PoseArray()
        msg_reeds.header.frame_id = "/map"
        msg_reeds.header.stamp = rospy.Time.now()

        for node in node_list:
            for px, py, pyaw in zip(node.path_x, node.path_y, node.path_yaw):
                msg_pose = Pose()
                # yaw to quaternion
                q = euler_to_quaternion(0, 0, pyaw)
                msg_pose.orientation.x = q[0]
                msg_pose.orientation.y = q[1]
                msg_pose.orientation.z = q[2]
                msg_pose.orientation.w = q[3]
                msg_pose.position.x = px
                msg_pose.position.y = py
                msg_reeds.poses.append(msg_pose)

        self.pub_reeds.publish(msg_reeds)

    def publish_final_reeds_shepp_path(self, path):
        # Publish wpt as MarkerArray
        msg_reeds = Marker()
        msg_reeds.header.frame_id= "/map"
        msg_reeds.header.stamp= rospy.Time.now()
        msg_reeds.ns= "spheres"
        msg_reeds.action= Marker.ADD
        msg_reeds.pose.orientation.w= 1.0

        msg_reeds.id = 0
        msg_reeds.type = Marker.SPHERE_LIST


        # POINTS markers use x and y scale for width/height respectively
        msg_reeds.scale.x = 0.1
        msg_reeds.scale.y = 0.1
        msg_reeds.scale.z = 0.1

        # Points are yellow
        msg_reeds.color.a = 1.0 # Don't forget to set the alpha!
        msg_reeds.color.r = 255/255.
        msg_reeds.color.g = 255/255.
        msg_reeds.color.b = 0/255.

        for node in path:
            # Tree
            if node.parent is not None:
                for x, y in zip(node.path_x, node.path_y):
                    msg_p = Point()
                    msg_p.x = x
                    msg_p.y = y
                    msg_reeds.points.append(msg_p)

        self.pub_final_reeds.publish(msg_reeds)

    def publish_final_reeds_shepp_path_posearray(self, path):
        # Publish final reeds_shepp as PoseArray
        msg_final_reeds = PoseArray()
        msg_final_reeds.header.frame_id = "/map"
        msg_final_reeds.header.stamp = rospy.Time.now()

        for node in path:
            for px, py, pyaw in zip(node.path_x, node.path_y, node.path_yaw):
                msg_pose = Pose()
                # yaw to quaternion
                q = euler_to_quaternion(0, 0, pyaw)
                msg_pose.orientation.x = q[0]
                msg_pose.orientation.y = q[1]
                msg_pose.orientation.z = q[2]
                msg_pose.orientation.w = q[3]
                msg_pose.position.x = px
                msg_pose.position.y = py
                msg_final_reeds.poses.append(msg_pose)

        self.pub_final_reeds.publish(msg_final_reeds)

def main():
    print("Start informed rrt star planning")
    start = [0.0, 0.0, np.deg2rad(0)]
    # goal  = [5.4, 3.2, np.deg2rad(90)]
    # goal  = [7.4, 3.1, np.deg2rad(90)] # world 1
    goal  = [7.4, 5.5, np.deg2rad(-90)] # world 2 (e_shape)
    # goal  = [7.4, 2.3, np.deg2rad(0)] # world 3 (vertical)
    randArea = [-3, 10]
    
    # Car params
    wheelbase             = 0.335
    steer_limit           = np.deg2rad(30)
    curvature_limit       = wheelbase / math.tan(steer_limit)
    # Reeds Shepp params
    reeds_shepp_step_size = 0.1 # 0.15

    agent = ReedsSheppRRTStar(start=start, goal=goal, randArea=randArea, eta=0.5, max_rewire_num=50, car_radius=0.25,
                            curvature_limit=curvature_limit, reeds_shepp_step_size=reeds_shepp_step_size)
        
    while not rospy.is_shutdown():
        agent.rate.sleep()


if __name__ == '__main__':
    main()
