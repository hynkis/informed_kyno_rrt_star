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

import parmap

import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

from plot_utils import Plot_utils

# ============================ #
# ----- Global functions ----- #
# ============================ #

def calc_dist(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None

class InformedRRTStar(Plot_utils):
    def __init__(self, start, goal, randArea, eta):
        """
            - start    : [start x, y]
            - goal     : [goal x, y]
            - randArea : [min_pose, max_pose]
        """
        # ROS init
        rospy.init_node('informed_rrt_star')

        # Parameters
        self.map = None
        self.obstacle_list = []
        self.node_list = None

        self.startNode = Node(start[0], start[1])
        self.goalNode  = Node(goal[0], goal[1])

        self.max_iter = 500
        self.goal_sample_rate = 10 # % of sampling goal point directly

        self.min_rand = randArea[0] # min random value
        self.max_rand = randArea[1] # max random value

        self.eta = eta  # expand distance

        # Subscriber
        self.sub_map = rospy.Subscriber("map", OccupancyGrid, self.callback_map)
        self.sub_init_pose = rospy.Subscriber("initialpose", PoseWithCovarianceStamped, self.callback_init_pose)
        self.sub_goal_pose = rospy.Subscriber("move_base_simple/goal", PoseStamped, self.callback_goal_pose)
        # Publisher
        self.pub_tree  = rospy.Publisher('/planner/informed_rrt_star', Path, queue_size=10)
        self.pub_local_obs = rospy.Publisher('/visual/obstacle', Marker, queue_size=10)
        self.pub_current_pose = rospy.Publisher('/visual/current_pose', Marker, queue_size=1)
        self.pub_goal_pose    = rospy.Publisher('/visual/goal_pose', Marker, queue_size=1)
        self.pub_node  = rospy.Publisher('/visual/node', Marker, queue_size=10)
        
        self.pub_tree = rospy.Publisher('/visual/tree', MarkerArray, queue_size=10)
        self.pub_ellipse = rospy.Publisher('/visual/ellipse', Marker, queue_size = 1)
        self.rate = rospy.Rate(100.0)

        # Map
        # - Initialize map
        while self.map is None:
            # Wait until map is initialized
            continue
        self.grid       = np.reshape(np.array(self.map.data), (self.map.info.width, self.map.info.height))
        self.resolution = self.map.info.resolution
        self.origin     = self.map.info.origin
        print("Origin :", self.origin)

        # Obstacle
        # curr_ix, curr_iy = self.get_map_index(2.4, 2.4)

        # obs_ix, obs_iy = np.where(self.grid[curr_ix-20:curr_ix+20, curr_iy-20:curr_iy+20] != 0)
        # for ix, iy in zip(obs_ix, obs_iy):
        #     self.obstacle_list.append(self.get_map_pose(ix + curr_ix-20, iy + curr_iy-20))
        # obs_ix, obs_iy = np.where(self.grid != 0)
        # for ix, iy in zip(obs_ix, obs_iy):
        #     self.obstacle_list.append(self.get_map_pose(obs_ix, obs_iy))
    
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
        self.goalNode.x = msg_pose.pose.position.x
        self.goalNode.y = msg_pose.pose.position.y
        
    # =============== #
    # ----- Map ----- #
    # =============== #

    def get_map_index(self, x, y):
        """
        Transform from x, y position in /map frame to numpy grid array x_index, y_index
        """ 
        x_ind = int((x-self.origin.position.x) // self.resolution)
        y_ind = int((y-self.origin.position.y) // self.resolution)
        # x, y are flipped in map_value
        return y_ind, x_ind

    def get_map_pose(self, ix, iy):
        """
        Transform from ix, iy to x, y
        """
        # x, y are flipped in map_value
        x = (iy+1) * self.resolution + self.origin.position.x
        y = (ix+1) * self.resolution + self.origin.position.y
        return x, y


    # ===================== #
    # ----- Functions ----- #
    # ===================== #

    def search(self):
        self.node_list = [self.startNode]
        cBest       = float('inf') # max length starts from inf
        rewire_cnt  = 0
        solutionSet = set()
        path        = None

        # Compute the sampling space
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
        M = a1 @ id1_t
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
            newNode = self.steer(nearestNode, rnd, self.eta)
            # Collision checking
            collision_free = self.check_collision_free(nearestNode, newNode)
            # Stretch
            if collision_free:
                # Find nearest node in tree
                near_inds = self.find_near_nodes(newNode)
                newNode   = self.choose_parent(newNode, near_inds)
                self.node_list.append(newNode)
                # Rewiring
                self.rewire(newNode, near_inds)
                if cBest < float('inf'):
                    rewire_cnt += 1

                # Goal check
                if self.is_near_goal(newNode):
                    # Check collision of the path from node to goal
                    if self.check_collision_free(newNode, self.goalNode):
                        solutionSet.add(newNode)
                        last_ind = len(self.node_list) - 1
                        tempPath = self.get_final_path(last_ind)
                        tempPath_len = self.get_path_len(tempPath)
                        if tempPath_len < cBest:
                            path  = tempPath
                            cBest = tempPath_len

                        self.publish_ellipse(xCenter, cBest, cMin, e_theta)

            # Visualization
            # - tree
            self.publish_point([(self.startNode.x, self.startNode.y)], publisher=self.pub_current_pose, color=(255,0,0))
            self.publish_point([(self.goalNode.x, self.goalNode.y)], publisher=self.pub_goal_pose, color=(0,0,255))
            self.publish_tree(self.node_list)

            print("For loop time :", time.time() - tic_for)
            if rewire_cnt > 30:
                print("DONE!!!")
                time.sleep(3)
                break

        return path


    def steer(self, nearNode, randNode, eta):
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

    def rewire(self, newNode, near_inds):
        """
        Calculate cost of near nodes and rewire to less-cost node
        Check whether
            cost(init --> newNode --> nearNode_i) < cost(init --> nearNode_i)
        if True
            - Update the parent of nearNode_i as newNode.
            - Update cost as (newNode.cost + dist)
        """
        n_node = len(self.node_list)
        for i in near_inds:
            nearNode = self.node_list[i]
            dist   = calc_dist(nearNode.x, nearNode.y, newNode.x, newNode.y)

            # Cost(init --> newNode --> nearNode_i)
            s_cost = newNode.cost + dist

            if s_cost < nearNode.cost:
                if self.check_collision_free(nearNode, newNode):
                    nearNode.parent = n_node - 1 # newNode become parent. (disconnected from past parent.)
                    nearNode.cost = s_cost

    # ==================== #
    # ----- Sampling ----- #
    # ==================== #

    def informed_sample(self, cMax, cMin, xCenter, C):
        # If a path to goal was found, sampling in an ellipsoid region
        print("cMax, cMin :", cMax, cMin)
        if cMax < float('inf'):
            r = [cMax*0.5,
                 math.sqrt(cMax**2 - cMin**2)*0.5,
                 math.sqrt(cMax**2 - cMin**2)*0.5]
            L = np.diag(r)
            xBall = self.sample_unit_ball()
            rnd = np.dot(np.dot(C, L), xBall) + xCenter # ellipsoid
            # rnd = [rnd[(0,0)], rnd[(1,0)]]
            rnd = Node(rnd[(0,0)], rnd[(1,0)]) # Node, not list

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
                       random.uniform(self.min_rand, self.max_rand))
        else:
            rnd = Node(self.goalNode.x, self.goalNode.y)

        return rnd


    # ================ #
    # ----- Node ----- #
    # ================ #

    def find_near_nodes(self, newNode):
        # Refer RRT*
        n_node = len(self.node_list)
        if n_node > 1:
            radius = 10.0 * math.sqrt((math.log(n_node) / n_node))
            d_list = [calc_dist(node.x, node.y, newNode.x, newNode.y) for node in self.node_list]
            near_inds = [d_list.index(i) for i in d_list if i <= radius**2]
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
        
        dist_list = []
        for i in near_inds:
            dist = calc_dist(self.node_list[i].x, self.node_list[i].y, newNode.x, newNode.y)
            collision_free = self.check_collision_free(self.node_list[i], newNode)
            if collision_free:
                dist_list.append(self.node_list[i].cost + dist)
            else:
                dist_list.append(float('inf'))

        min_cost = min(dist_list)
        min_ind  = near_inds[dist_list.index(min_cost)]

        if min_cost == float('inf'):
            print("Min cost is inf")
            return newNode
        
        # Wiring min cost node with newNode as parent 
        newNode.cost = min_cost
        newNode.parent = min_ind

        print("choose parent time :", time.time() - tic)

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


    def is_near_goal(self, node):
        """
        Check goal arrival
        """
        dist = calc_dist(node.x, node.y, self.goalNode.x, self.goalNode.y)
        if dist < self.eta:
            return True
        return False


    # ================ #
    # ----- Path ----- #
    # ================ #
    def get_final_path(self, last_ind):
        """
        Backpropagate solution path
        Path is the list of Node
        """
        path = [self.goalNode]
        while self.node_list[last_ind].parent is not None:
            node = self.node_list[last_ind]
            path.append(node)
            last_ind = node.parent
        path.append(self.startNode)

        return path

    def get_path_len(self, path):
        path_len = 0
        num_node = len(path)
        for i in range(1, num_node):
            path_len += calc_dist(path[i].x, path[i].y, path[i-1].x, path[i-1].y)

        return path_len


    # ===================== #
    # ----- Collision ----- #
    # ===================== #

    def check_collision_free(self, fromNode, toNode):
        """
        Check collision free from 'fromNode' to 'toNode'
        """
        tmpNode = copy.deepcopy(fromNode)
        dist = calc_dist(fromNode.x, fromNode.y, toNode.x, toNode.y)
        dist_res = self.resolution*0.2
        d = dist_res
        while d < dist:
            # Steer
            tmpNode = self.steer(tmpNode, toNode, dist_res)
            # Get grid value
            ix, iy = self.get_map_index(tmpNode.x, tmpNode.y)
            if self.grid[ix, iy] != 0:
                # print("Collision :", ix, iy, self.grid[ix, iy])
                return False # collision
            d += dist_res

        return True

    # ======================= #
    # ----- Visulalizer ----- #
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
        msg_point.color.a = 1.0; # Don't forget to set the alpha!
        msg_point.color.r = color[0]/255.;
        msg_point.color.g = color[1]/255.;
        msg_point.color.b = color[2]/255.;

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
        msg_point.color.a = 1.0; # Don't forget to set the alpha!
        msg_point.color.r = 0/255.;
        msg_point.color.g = 255/255.;
        msg_point.color.b = 0/255.;

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

                parentNode = self.node_list[node.parent]
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

        self.pub_tree.publish(msg_tree)
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
        fx = R @ np.array([x, y])
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


def main():
    print("Start informed rrt star planning")
    start = [0, 0]
    goal  = [5.4, 3.2] # goal  = [5.4, 3.2, np.deg2rad(90)]
    randArea = [-2, 9]
    agent = InformedRRTStar(start=start, goal=goal, randArea=randArea, eta=0.5)

    # create obstacles
    obstacleList = [
        (1.40, 3.00, 0.8),
        (2.30, 2.17, 0.8),
        (3.35, 2.10, 0.8),
        (4.45, 2.15, 0.8),
        (4.40, 3.20, 0.8),
    ]

    # # Set params
    # show_animation = False

    # rrt = InformedRRTStar(start=[0, 0], goal=[2.4, 3.11],
    #                       randArea=[-2, 9], obstacleList=obstacleList)
    # tic = time.time()
    # path = rrt.informed_rrt_star_search(animation=show_animation)
    # print("Done!! time :", time.time() - tic)


    # # Plot path
    # # if show_animation:
    # rrt.draw_graph()
    # plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    # plt.grid(True)
    # plt.pause(0.0001) # 0.01
    # plt.show()
        
    while not rospy.is_shutdown():

        # Test collision checking
        # collision_free = agent.check_collision_free(agent.startNode, agent.goalNode)
        # print("Collision free :", collision_free)

        # Sampling
        agent.search()
        agent.rate.sleep()


if __name__ == '__main__':
    main()
