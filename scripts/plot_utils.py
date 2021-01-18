#!/usr/bin/env python

import rospy
import math
import os
import sys
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped

class Plot_utils:
    
    def draw_graph(self):
        pass

    def plot_path(self, node_list, color=(255, 0, 0)):
        tree = MarkerArray()
        id = 1
        for node in node_list:
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
                path.color.r = color[0]/255.
                path.color.g = color[1]/255.
                path.color.b = color[2]/255.

                path.lifetime = rospy.Duration()
                path.pose.orientation.w = 1.0

                p1 = Point()
                p1.x = node.parent.x
                p1.y = node.parent.y
                p1.z = 0.02
                path.points.append(p1)

                p2 = Point()
                p2.x = node.x
                p2.y = node.y
                p2.z = 0.02
                path.points.append(p2)

                tree.markers.append(path)

        self.pub_tree.publish(tree)
                
