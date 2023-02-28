#!/usr/bin/env python
import math
from math import *
import json
import copy
import argparse

import rospy
import numpy as np
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from const import *

from planner.msg import ContinuousAction, DiscreteAction, ContinuousState, DiscreteState
from planner.srv import (
    DiscreteActionSequenceExec,
    ContinuousActionSequenceExec,
    DiscreteActionStochasticExec,
)

from src.planner.src.base_planner1 import Planner, RobotClient

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit


def dump_action_table(action_table, filename):
    """dump the MDP policy into a json file

    Arguments:
        action_table {dict} -- your mdp action table. It should be of form {(1,2,0): (1, 0), ...}
        filename {str} -- output filename
    """
    tab = dict()
    for k, v in action_table.items():
        key = [str(int(i)) for i in k]
        key = ",".join(key)
        tab[key] = v

    with open(filename, 'w') as fout:
        json.dump(tab, fout)


class DSDAPlanner(Planner):


    def setup_map(self):
        """Get the occupancy grid and inflate the obstacle by some pixels.

        You should implement the obstacle inflation yourself to handle uncertainty.
        """
        # Hint: search the ROS message defintion of OccupancyGrid
        occupancy_grid = rospy.wait_for_message('/map', OccupancyGrid)
        self.map = occupancy_grid.data
        # you should inflate the map to get self.aug_map
        self.aug_map = copy.deepcopy(self.map)

        ####################### TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask
        # print out self.map to see what the data format is like
        # int8[] array
        # neighboring nodes whose rounded up to integer Euclidean distance to current center node is less than or equal to 3
        self.aug_map = list(self.aug_map)
        nei_relative_position = []
        def euclidean_distance_to_center(x, y):
            return np.round(np.sqrt(x**2 + y**2))
        for x in range(-self.inflation_ratio, self.inflation_ratio + 1):
            for y in range(-self.inflation_ratio, self.inflation_ratio + 1):
                if x == 0 and y == 0:
                    continue
                if euclidean_distance_to_center(x, y) <= inflation_ratio:
                    nei_relative_position.append([x, y])

        # when inflation radius is 3
        # nei_relative_position = [[3, 0], [3, 1], [2, 2], [1, 3], [0, 3], [-1, 3],
        #                          [-2, -2], [-3, 1], [-3, 0], [-3, -1], [-2, 2],
        #                          [-1, -3], [0, -3], [1, -3], [2, -2], [3, -1],
        #                          [2, -1], [2, 0], [2, 1], [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
        #                          [0, -2], [0, -1], [0, 1], [0, 2],
        #                          [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
        #                          [-2, -1], [-2, 0], [-2, 1]]

        for x in range(self.world_width):
            for y in range(self.world_height):
                # get neighbor position
                for nei_relative_x, nei_relative_y in nei_relative_position:
                    nei_x, nei_y = x + nei_relative_x, y + nei_relative_y
                    # if neighboring node not within map boundary, then skip
                    if not (0 <= nei_x < self.world_width
                            and 0 <= nei_y < self.world_height):
                        continue
                    # update neighbor value to be max(center current position occupancy value, neighbor occupancy value)
                    nei_val = self.aug_map[self.xy_to_1d_grid_index(nei_x, nei_y)]
                    center_val = self.aug_map[self.xy_to_1d_grid_index(x, y)]

                    self.aug_map[self.xy_to_1d_grid_index(nei_x, nei_y)] = max(nei_val, center_val)

        self.aug_map = tuple(self.aug_map)
        ###################################<- end of FILL ME

    def xy_to_1d_grid_index(self, x, y):
        return y * self.world_width + x

    def generate_plan(self, init_pose):
        """TODO: FILL ME! This function generates the plan for the robot, given
        an initial pose and a goal pose.

        You should store the list of actions into self.action_seq, or the policy
        into self.action_table.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """

        ############################################-> start of FILL ME!

        #### for DSDA
        # A* algorithm implementation, assuming discrete states
        ## Manhattan distance as the heuristic H
        def h_manhattan(x1, y1):
            x2, y2 = self._get_goal_position()
            return abs(x1 - x2) + abs(y1 - y2)

        from queue import PriorityQueue
        init_x, init_y, init_theta = init_pose
        frontier = PriorityQueue()
        frontier.put((0, (init_x, init_y, init_theta)))  # cost to come, priority queue: point location (x, y, theta)


        g_score = {(init_x, init_y): 0}
        f_score = {(init_x, init_y): h_manhattan(init_x, init_y)}
        path_parent = dict()

        while frontier.not_empty:
            current_f, (current_x, current_y) = frontier.get()

            if (current_x, current_y) == self._get_goal_position():
                break

            for next_relative_x, next_relative_y in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
                nei_x = next_relative_x + current_x
                nei_y = next_relative_y + current_y
                # make sure next x, y is within boundary and occupancy rate is below 100
                if not (0 <= nei_x < self.world_width
                        and 0 <= nei_y < self.world_height):
                    continue
                if self.collision_checker(nei_x, nei_y) >= 100:
                    continue

                tmp_nei_g_score = g_score[(current_x, current_y)] + 1
                tmp_nei_f_score = tmp_nei_g_score + h_manhattan(nei_x, nei_y)

                if tmp_nei_f_score < f_score[(nei_x, nei_y)] or (nei_x, nei_y) not in f_score:
                    g_score[(nei_x, nei_y)] = tmp_nei_g_score
                    f_score[(nei_x, nei_y)] = tmp_nei_f_score
                    frontier.put((tmp_nei_f_score, (nei_x, nei_y)))
                    path_parent[(nei_x, nei_y)] = (current_x, current_y)

        # get current path
        path_seq_rev = [self._get_goal_position()]
        cur_x, cur_y = self._get_goal_position()
        while (cur_x, cur_y) != (init_x, init_y):
            (cur_x, cur_y) = path_parent[(cur_x, cur_y)]
            path_seq_rev.append((cur_x, cur_y))

        path_seq_rev.reverse()
        path_seq = path_seq_rev

        # get action sequence
        self.action_seq = []
        cur_x, cur_y, cur_ori = init_pose
        cur_step = 0
        while cur_step < len(path_seq) - 1:  # hasn't arrived at goal, needs one more step
            next_x, next_y = path_seq[cur_step + 1]
            ## forward action
            if (cur_ori == 0 and (next_x - cur_x, next_y - cur_y) == (1, 0)) \
                    or (cur_ori == 1 and (next_x - cur_x, next_y - cur_y) == (0, 1)) \
                    or (cur_ori == 2 and (next_x - cur_x, next_y - cur_y) == (-1, 0)) \
                    or (cur_ori == 3 and (next_x - cur_x, next_y - cur_y) == (0, -1)):
                self.action_seq.append((1, 0))
                cur_x, cur_y, cur_ori = next_x, next_y, cur_ori

            ## turn left then forward
            elif (cur_ori == 0 and (next_x - cur_x, next_y - cur_y) == (0, 1)) \
                    or (cur_ori == 1 and (next_x - cur_x, next_y - cur_y) == (-1, 0)) \
                    or (cur_ori == 2 and (next_x - cur_x, next_y - cur_y) == (0, -1)) \
                    or (cur_ori == 3 and (next_x - cur_x, next_y - cur_y) == (1, 0)):
                self.action_seq.append((0, 1))
                self.action_seq.append((1, 0))
                cur_x, cur_y, cur_ori = next_x, next_y, (cur_ori + 1) % 4

            ## turn right then forward
            elif (cur_ori == 0 and (next_x - cur_x, next_y - cur_y) == (0, -1)) \
                    or (cur_ori == 1 and (next_x - cur_x, next_y - cur_y) == (1, 0)) \
                    or (cur_ori == 2 and (next_x - cur_x, next_y - cur_y) == (0, 1)) \
                    or (cur_ori == 3 and (next_x - cur_x, next_y - cur_y) == (-1, 0)):
                self.action_seq.append((0, -1))
                self.action_seq.append((1, 0))
                cur_x, cur_y, cur_ori = next_x, next_y, (cur_ori - 1) % 4

            ## turn left then turn left then forward
            elif (cur_ori == 0 and (next_x - cur_x, next_y - cur_y) == (-1, 0)) \
                    or (cur_ori == 1 and (next_x - cur_x, next_y - cur_y) == (0, -1)) \
                    or (cur_ori == 2 and (next_x - cur_x, next_y - cur_y) == (1, 0)) \
                    or (cur_ori == 3 and (next_x - cur_x, next_y - cur_y) == (0, 1)):
                self.action_seq.append((0, 1))
                self.action_seq.append((0, 1))
                self.action_seq.append((1, 0))
                cur_x, cur_y, cur_ori = next_x, next_y, (cur_ori + 2) % 4

            cur_step += 1


        # #### for CSDA hybrid A*
        # # hybrid A* algorithm implementation, assuming continuous states
        # ## Euclidean distance as the heuristic H
        # def h_euclidean(x1, y1):
        #     x2, y2 = self._get_goal_position()
        #     return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # def pose_is_close_to_goal(x, y):
        #     return math.sqrt( (x - self._get_goal_position()[0]) ** 2
        #                       + (y - self._get_goal_position()[1]) ** 2 ) < 0.3
        #
        # init_x, init_y, init_theta = init_pose
        # frontier = PriorityQueue()
        # frontier.put((0, (init_x, init_y, init_theta)))  # cost to come, priority queue: point location (x, y, theta)
        #
        # g_score = {(init_x, init_y): 0}
        # f_score = {(init_x, init_y): h_euclidean(init_x, init_y)}
        # path_parent = dict()
        # path_control_from_parent = dict()
        #
        # while frontier.not_empty:
        #     current_f, (current_x, current_y, current_theta) = frontier.get()
        #
        #     if pose_is_close_to_goal(current_x, current_y):
        #         path_seq_rev = [(current_x, current_y, current_theta)]
        #         break
        #
        #     # sample a neighboring node that can be reached within one timestep
        #     for _ in range(10):
        #         # uniform sample v and w
        #         v = np.random.uniform(0, 1)
        #         w = np.random.uniform(-pi, pi)
        #
        #         nei_x, nei_y, nei_theta = self.motion_predict(cur_x, cur_y, current_theta, v, w)
        #         # make sure next x, y is within boundary and occupancy rate is below 100
        #         if not (0 <= nei_x < self.world_width
        #                 and 0 <= nei_y < self.world_height):
        #             continue
        #         if self.collision_checker(nei_x, nei_y) >= 100:
        #             continue
        #
        #         tmp_nei_g_score = g_score[(current_x, current_y)] + v / w * abs(nei_theta - current_theta)
        #         tmp_nei_f_score = tmp_nei_g_score + h_euclidean(nei_x, nei_y)
        #         nei_x_grid_index = int(nei_x)
        #         nei_y_grid_index = int(nei_y)
        #
        #         if tmp_nei_f_score < f_score[(nei_x_grid_index, nei_y_grid_index)] \
        #                 or (nei_x_grid_index, nei_y_grid_index) not in f_score:
        #             g_score[(nei_x_grid_index, nei_y_grid_index)] = tmp_nei_g_score
        #             f_score[(nei_x_grid_index, nei_y_grid_index)] = tmp_nei_f_score
        #             frontier.put((tmp_nei_f_score, (nei_x, nei_y, nei_theta)))
        #             path_parent[(nei_x, nei_y, nei_theta)] = (current_x, current_y, current_theta)
        #             path_control_from_parent[(nei_x, nei_y, nei_theta)] = (v, w)
        #
        # # get action sequence according to sequence path tree
        # self.action_seq = []
        # while (current_x, current_y) != (init_x, init_y):
        #     self.action_seq.append(path_control_from_parent[(current_x, current_y, current_theta)])
        #     (current_x, current_y, current_theta) = path_parent[(current_x, current_y, current_theta)]
        #
        # self.action_seq.reverse()


        ### for DSPA
        ########### save action table for DSPA




    def collision_checker(self, x, y):
        """TODO: FILL ME!
        You should implement the collision checker.
        Hint: you should consider the augmented map and the world size
        
        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
        
        Returns:
            bool -- True for collision, False for non-collision
        """
        return (0 <= x < self.world_width / self.resolution and 0 <= y < self.world_height / self.resolution) \
            and self.aug_map[self.xy_to_1d_grid_index(x, y)] == 100


if __name__ == "__main__":
    # You can generate and save the plan using the code below
    rospy.init_node('planner')
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', type=str, default='1,8',
                        help='goal position')
    parser.add_argument('--com', type=int, default=0,
                        help="if the map is com1 map")
    args = parser.parse_args()

    try:
        goal = [int(pose) for pose in args.goal.split(',')]
    except:
        raise ValueError("Please enter correct goal format")

    if args.com:
        width = 2500
        height = 983
        resolution = 0.02
    else:
        width = 200
        height = 200
        resolution = 0.05

    robot = RobotClient()
    inflation_ratio = 3  # TODO: You should change this value accordingly
    planner = Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan(robot.get_current_discrete_state())

    # ##################
    # # rospy.init_node("lab1_robot_interface")
    # # interface = Lab1Interface()
    # # rospy.loginfo("Robot action interface ready!")
    # topic = 'chatter'
    # pub = rospy.Publisher(topic, String)
    # rospy.init_node('talker', anonymous=True)
    # rospy.loginfo("I will publish to the topic %s", topic)
    # while not rospy.is_shutdown():
    #     str = "hello world %s" % rospy.get_time()
    #     # str = self.map
    #     rospy.loginfo(str)
    #     pub.publish(str)
    #     rospy.sleep(0.1)
    #
    # ##################

    # Let's try executing a hard-coded motion plan!
    # Forward! Forward! Turn left!
    # You should see an AssertionError since we didn't reach the goal.
    mock_action_plan = [(1, 0), (1, 0), (0, 1), (0, -1), (1, 0), (1, 0)]
    robot.publish_discrete_control(mock_action_plan, goal)
    # TODO: FILL ME!
    # After you implement your planner, send the plan/policy generated by your
    # planner using appropriate execution calls.
    # e.g., robot.publish_discrete_control(planner.action_seq, goal)

    # save your action sequence
    result = np.array(planner.action_seq)
    np.savetxt("actions_continuous.txt", result, fmt="%.2e")

    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')
