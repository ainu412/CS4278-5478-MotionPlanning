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

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit

### move to new class

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


class Planner:
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        """init function of the base planner. You should develop your own planner
        using this class as a base.

        For standard mazes, width = 200, height = 200, resolution = 0.05.
        For COM1 map, width = 2500, height = 983, resolution = 0.02

        Arguments:
            world_width {int} -- width of map in terms of pixels
            world_height {int} -- height of map in terms of pixels
            world_resolution {float} -- resolution of map

        Keyword Arguments:
            inflation_ratio {int} -- [description] (default: {3})
        """
        self.map = None
        self.pose = None
        self.goal = None
        self.action_seq = None  # output
        self.aug_map = None  # occupancy grid with inflation
        self.action_table = {}



        self.world_width = world_width
        self.world_height = world_height
        self.resolution = world_resolution

        self.unit_width = int(world_width * world_resolution)
        self.unit_height = int(world_height * world_resolution)

        self.inflation_ratio = inflation_ratio
        self.setup_map()
        rospy.sleep(1)

    def setup_map(self):
        """Get the occupancy grid and inflate the obstacle by some pixels.

        You should implement the obstacle inflation yourself to handle uncertainty.
        """
        # Hint: search the ROS message defintion of OccupancyGrid
        occupancy_grid = rospy.wait_for_message('/map', OccupancyGrid)
        self.map = occupancy_grid.data

        # TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask

        # you should inflate the map to get self.aug_map
        self.aug_map = copy.deepcopy(self.map)

    def _get_goal_position(self):
        goal_position = self.goal.pose.position
        return (goal_position.x, goal_position.y)

    def set_goal(self, x, y, theta=0):
        """set the goal of the planner

        Arguments:
            x {int} -- x of the goal
            y {int} -- y of the goal

        Keyword Arguments:
            theta {int} -- orientation of the goal; we don't consider it in our planner (default: {0})
        """
        a = PoseStamped()
        a.pose.position.x = x
        a.pose.position.y = y
        a.pose.orientation.z = theta
        self.goal = a

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
        self.action_seq = []

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

        return False

    def motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """Predict the next pose of the robot given controls. Returns None if
        the robot collide with the wall.

        The robot dynamics is provided in the assignment description.

        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
            theta {float} -- current theta of robot
            v {float} -- linear velocity
            w {float} -- angular velocity

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        num_steps = int(dt * frequency)
        dx = 0
        dy = 0
        for i in range(num_steps):
            if w != 0:
                dx = - v / w * np.sin(theta) + v / w * \
                     np.sin(theta + w / frequency)
                dy = v / w * np.cos(theta) - v / w * \
                     np.cos(theta + w / frequency)
            else:
                dx = v * np.cos(theta) / frequency
                dy = v * np.sin(theta) / frequency
            x += dx
            y += dy

            if not (0 <= x < self.unit_width and 0 <= y < self.unit_height):
                return None
            if self.collision_checker(int(x / self.resolution), int(y / self.resolution)):
                return None
            theta += w / frequency
        return x, y, theta

    def discrete_motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """Discrete version of the motion predict.

        Note that since the ROS simulation interval is set to be 0.5 sec and the
        robot has a limited angular speed, to achieve 90 degree turns, we have
        to execute two discrete actions consecutively. This function wraps the
        discrete motion predict.

        Please use it for your discrete planner.

        Arguments:
            x {int} -- current x of robot
            y {int} -- current y of robot
            theta {int} -- current theta of robot
            v {int} -- linear velocity
            w {int} -- angular velocity (0, 1, 2, 3)

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision or out of boundary
        """
        w_radian = w * np.pi / 2
        first_step = self.motion_predict(x, y, theta * np.pi / 2, v, w_radian)
        if first_step:
            second_step = self.motion_predict(
                first_step[0], first_step[1], first_step[2], v, w_radian)
            if second_step:
                return (round(second_step[0]), round(second_step[1]), round(second_step[2] / (np.pi / 2)) % 4)
        return None


class RobotClient:
    """A class to interface with the (simulated) robot.

    You can think of this as the "driver" program provided by the robot manufacturer ;-)
    """

    def __init__(self):
        self._cstate = None
        self.sb_cstate = rospy.Subscriber(
            "/lab1/continuous_state", ContinuousState, self._cstate_callback
        )
        self._dstate = None
        self.sb_dstate = rospy.Subscriber(
            "/lab1/discrete_state", DiscreteState, self._dstate_callback
        )

    def _cstate_callback(self, msg):
        """Callback of the subscriber."""
        self._cstate = msg

    def get_current_continuous_state(self):
        """Get the current continuous state.

        Returns:
            tuple -- x, y, \theta, as defined in the instruction document.
        """
        return (self._cstate.x, self._cstate.y, self._cstate.theta)

    def _dstate_callback(self, msg):
        self._dstate = msg

    def get_current_discrete_state(self):
        """Get the current discrete state.

        Returns:
            tuple -- x, y, \theta, as defined in the instruction document.
        """
        return (self._dstate.x, self._dstate.y, self._dstate.theta)

    def _d_from_target(self, target_pose):
        """Compute the distance from current pose to the target_pose.

        Arguments:
            pose {list} -- robot pose

        Returns:
            float -- distance to the target_pose
        """
        pose = self.get_current_continuous_state()
        return math.sqrt(
            (pose[0] - target_pose[0]) ** 2 + (pose[1] - target_pose[1]) ** 2
        )

    def is_close_to_goal(self, goal):
        """Check if close enough to the given goal.

        Arguments:
            pose {list} -- robot post

        Returns:
            bool -- goal or not
        """
        return self._d_from_target(goal) < 0.3

    def publish_discrete_control(self, action_seq, goal):
        """Publish the discrete controls"""
        proxy = rospy.ServiceProxy(
            "/lab1/discrete_action_sequence",
            DiscreteActionSequenceExec,
        )
        plan = [DiscreteAction(action) for action in action_seq]
        proxy(plan)
        assert self.is_close_to_goal(goal), "Didn't reach the goal."

    def publish_discrete_control_one(self, action):
        """Publish the discrete controls"""
        proxy = rospy.ServiceProxy(
            "/lab1/discrete_action_sequence",
            DiscreteActionSequenceExec,
        )
        plan = [DiscreteAction(action)]
        proxy(plan)
        # assert self.is_close_to_goal(goal), "Didn't reach the goal."

    def publish_continuous_control(self, action_seq, goal):
        """Publish the continuous controls.

        TODO: FILL ME!

        You should implement the ROS service request to execute the motion plan.

        The service name is /lab1/continuous_action_sequence

        The service type is ContinuousActionSequenceExec

        Checkout the definitions in planner/msg/ and planner/srv/
        """
        pass
        assert self.is_close_to_goal(goal)

    def publish_continuous_control_one(self, action):
        """Publish the continuous controls.

        TODO: FILL ME!

        You should implement the ROS service request to execute the motion plan.

        The service name is /lab1/continuous_action_sequence

        The service type is ContinuousActionSequenceExec

        Checkout the definitions in planner/msg/ and planner/srv/
        """
        proxy = rospy.ServiceProxy(
            "/lab1/continuous_action_sequence",
            ContinuousActionSequenceExec,
        )
        plan = [ContinuousAction(action[0], action[1])]
        proxy(plan)

    def execute_policy(self, action_table, goal):
        """Execute a given policy in MDP.

        Due to the stochastic dynamics, we cannot execute the motion plan
        without feedback. Hence, every time we execute a discrete action, we
        query the current state by `get_current_discrete_state()`.

        You don't have to worry about the stochastic dynamics; it is implemented
        in the simulator. You only need to send the discrete action.
        """
        # TODO: FILL ME!
        # Instantiate the ROS service client
        # Service name: /lab1/discrete_action_stochastic
        # Service type: DiscreteActionStochasticExec
        # Checkout the definitions in planner/msg/ and planner/srv/
        while not self.is_close_to_goal(goal):
            current_state = self.get_current_discrete_state()
            action = action_table[current_state]
            # TODO: FILL ME!
            # Put the action into proper ROS request and send it
        rospy.sleep(1)
        assert self.is_close_to_goal(goal)


class DSPAPlanner(Planner):
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3,
                 max_iteration=3, discount_factor=0.9, converge_threshold=0.1):
        """init function of the base planner. You should develop your own planner
        using this class as a base.

        For standard mazes, width = 200, height = 200, resolution = 0.05.
        For COM1 map, width = 2500, height = 983, resolution = 0.02

        Arguments:
            world_width {int} -- width of map in terms of pixels
            world_height {int} -- height of map in terms of pixels
            world_resolution {float} -- resolution of map

        Keyword Arguments:
            inflation_ratio {int} -- [description] (default: {3})
        """
        self.map = None
        self.pose = None
        self.goal = None
        self.action_seq = None  # output
        self.aug_map = None  # occupancy grid with inflation
        self.action_table = {}

        self.unit_width = int(world_width * world_resolution)
        self.unit_height = int(world_height * world_resolution)

        self.world_width = world_width
        self.world_height = world_height
        self.resolution = world_resolution

        ######### ->newly added for DSPAPlanne
        self.states = None
        self.max_iteration = max_iteration
        self.discount_factor = discount_factor
        self.converge_threshold = converge_threshold
        self.utility = None
        ######### <-

        self.inflation_ratio = inflation_ratio
        self.setup_map()
        rospy.sleep(1)

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
            return np.round(np.sqrt(x ** 2 + y ** 2))

        for x in range(-self.inflation_ratio, self.inflation_ratio + 1):
            for y in range(-self.inflation_ratio, self.inflation_ratio + 1):
                # print('x, y, euclidean', x, y, euclidean_distance_to_center(x, y))
                if x == 0 and y == 0: # skip center
                    continue
                if euclidean_distance_to_center(x, y) <= self.inflation_ratio:
                    nei_relative_position.append([x, y])
        # print('self.inflation_ratio', self.inflation_ratio)
        # print('nei_relative_position', nei_relative_position)
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
                if not self.collision_checker_wrt_original_map(x, y):
                    continue

                # get neighbor position
                for nei_relative_x, nei_relative_y in nei_relative_position:
                    nei_x, nei_y = x + nei_relative_x, y + nei_relative_y
                    # if neighboring node not within map boundary, then skip
                    if not (0 <= nei_x < self.world_width
                            and 0 <= nei_y < self.world_height):
                        continue
                    # update neighbor value to be max(center current position occupancy value, neighbor occupancy value)
                    self.aug_map[self.xy_to_1d_grid_index(nei_x, nei_y)] = 100

        self.aug_map = tuple(self.aug_map)

        # visualize non aug map
        # for y in range(199, -1 , -1):
        #     print("".join(['+' if self.map[self.xy_to_1d_grid_index(x, y)] == 100 else ' ' for x in range(200)]))
        # visualize aug map DONE!
        # for y in range(199, -1 , -1):
        #     print("".join(['+' if self.aug_map[self.xy_to_1d_grid_index(x, y)] == 100 else ' ' for x in range(200)]))

        # all non occupied (free) grid are possible state
        self.states = []
        for x_unit in range(self.unit_width):
            for y_unit in range(self.unit_height):
                if not self.collision_checker(int(x_unit / self.resolution), int(y_unit / self.resolution)):
                    for theta in [0, 1, 2, 3]:
                        self.states.append((x_unit, y_unit, theta))
        print('states', self.states)
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
        def reward_func(x_unit, y_unit):
            # goal
            if (x_unit, y_unit) == self._get_goal_position():
                return 100
            # occupied state, can set reward according to occupancy rate; but now simplified
            # elif self.collision_checker(int(x_unit / self.resolution), int(y_unit / self.resolution)):
            #     return -100
            # free state
            return -50

        # utility initialization for all states
        utility = dict()
        for s in self.states:
            utility[s] = 0

        # compute utility for each state
        for _ in range(self.max_iteration):
            delta = 0.0
            for x_unit, y_unit, theta in self.states:
                # terminal state, goal state
                if (x_unit, y_unit) == self._get_goal_position():
                    continue

                q = {'FORWARD': -float("inf"), 'LEFT': -float("inf"), 'RIGHT': -float("inf"), 'STAY': -float("inf")}
                for action in ['FORWARD', 'LEFT', 'RIGHT', 'STAY']:
                    if action == 'LEFT':
                        nei_pose = self.discrete_motion_predict(x_unit, y_unit, theta, 0, 1)
                        if nei_pose is None:
                            continue
                        nei_x_unit, nei_y_unit, nei_theta = nei_pose
                        reward = reward_func(nei_x_unit, nei_y_unit)
                        q['LEFT'] = reward + self.discount_factor * utility[nei_pose]
                    elif action == 'RIGHT':
                        nei_pose = self.discrete_motion_predict(x_unit, y_unit, theta, 0, -1)
                        if nei_pose is None:
                            continue
                        nei_x_unit, nei_y_unit, nei_theta = nei_pose
                        reward = reward_func(nei_x_unit, nei_y_unit)
                        q['RIGHT'] = reward + self.discount_factor * utility[nei_pose]
                    elif action == 'STAY':
                        nei_pose = x_unit, y_unit, theta
                        nei_x_unit, nei_y_unit, nei_theta = nei_pose
                        reward = reward_func(nei_x_unit, nei_y_unit)
                        q['STAY'] = reward + self.discount_factor * utility[nei_pose]
                    elif action == 'FORWARD':
                        q_sum = 0
                        has_nei_state = False
                        for v, w, p in [(1, 0, 0.9), (pi/2, 1, 0.05), (pi/2, -1, 0.05)]:
                            nei_pose = self.discrete_motion_predict(x_unit, y_unit, theta, v, w)
                            if nei_pose is None:
                                continue

                            has_nei_state = True
                            nei_x_unit, nei_y_unit, nei_theta = nei_pose
                            if not (0 <= nei_x_unit < self.unit_width and 0 <= nei_y_unit < self.unit_height):
                                return None
                            reward = reward_func(nei_x_unit, nei_y_unit)
                            q_sum += p * (reward + self.discount_factor * utility[nei_pose])
                        if has_nei_state:
                            q['FORWARD'] = q_sum
                tmp_utility = max(q.values())
                delta = max(0, abs(tmp_utility - utility[x_unit, y_unit, theta]))
                utility[x_unit, y_unit, theta] = tmp_utility

            if delta < self.converge_threshold:
                break

        # compute optimal policy for each state
        self.action_table = dict()
        print(11)
        for x_unit, y_unit, theta in self.states:
            print(0)
            q = {'FORWARD': -float("inf"), 'LEFT': -float("inf"), 'RIGHT': -float("inf"), 'STAY': -float("inf")}
            for action in ['FORWARD', 'LEFT', 'RIGHT', 'STAY']:
                if action == 'LEFT':
                    nei_pose = self.discrete_motion_predict(x_unit, y_unit, theta, 0, 1)
                    if nei_pose is None:
                        continue
                    nei_x_unit, nei_y_unit, nei_theta = nei_pose
                    reward = reward_func(nei_x_unit, nei_y_unit)
                    q['LEFT'] = reward + self.discount_factor * utility[nei_pose]
                elif action == 'RIGHT':
                    nei_pose = self.discrete_motion_predict(x_unit, y_unit, theta, 0, -1)
                    if nei_pose is None:
                        continue
                    nei_x_unit, nei_y_unit, nei_theta = nei_pose
                    reward = reward_func(nei_x_unit, nei_y_unit)
                    q['RIGHT'] = reward + self.discount_factor * utility[nei_pose]
                elif action == 'STAY':
                    nei_pose = x_unit, y_unit, theta
                    nei_x_unit, nei_y_unit, nei_theta = nei_pose
                    reward = reward_func(nei_x_unit, nei_y_unit)
                    q['STAY'] = reward + self.discount_factor * utility[nei_pose]
                elif action == 'FORWARD':
                    q_sum = 0
                    has_nei_state = False
                    for v, w, p in [(1, 0, 0.9), (pi / 2, 1, 0.05), (pi / 2, -1, 0.05)]:
                        nei_pose = self.discrete_motion_predict(x_unit, y_unit, theta, v, w)
                        if nei_pose is None:
                            continue
                        has_nei_state = True
                        nei_x_unit, nei_y_unit, nei_theta = nei_pose
                        if not (0 <= nei_x_unit < self.unit_width and 0 <= nei_y_unit < self.unit_height):
                            return None

                        reward = reward_func(nei_x_unit, nei_y_unit)
                        q_sum += p * (reward + self.discount_factor * utility[nei_pose])
                    if has_nei_state:
                        q['FORWARD'] = q_sum
            # string to action list[2]
            action_str2li = {'LEFT': [0, 1], 'RIGHT': [0, -1], 'FORWARD': [1, 0], 'STAY': [0, 0]}

            self.action_table[(x_unit, y_unit, theta)] = action_str2li[max(q, key=q.get)]


        self.utility = utility


    def collision_checker(self, x, y):
        """TODO: FILL ME!
        You should implement the collision checker.
        Hint: you should consider the augmented map and the world size

        Arguments:
            x {int} -- current x of robot
            y {int} -- current y of robot

        Returns:
            bool -- True for collision, False for non-collision
        """
        # print('augmap[400:600]', self.aug_map[400:600])
        # print('x', x, 'y', 'y')
        # print('self.xy_to_1d_grid_index(x, y)', self.xy_to_1d_grid_index(x, y))
        # print('self.aug_map[self.xy_to_1d_grid_index(x, y)]', self.aug_map[self.xy_to_1d_grid_index(x, y)])
        return (0 <= x < self.world_width and 0 <= y < self.world_height) \
               and self.aug_map[self.xy_to_1d_grid_index(x, y)] == 100

    def collision_checker_wrt_original_map(self, x, y):
        return (0 <= x < self.world_width and 0 <= y < self.world_height) \
               and self.map[self.xy_to_1d_grid_index(x, y)] == 100


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
    inflation_ratio = 16  # TODO: You should change this value accordingly
    planner = DSPAPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])


    planner.generate_plan(robot.get_current_discrete_state())

    # i = 0
    # compute action sequence according to policy
    # while robot.get_current_discrete_state()[:2] != planner._get_goal_position():
    #     cur_loc = robot.get_current_discrete_state()
    #     nominal_action = planner.action_table[cur_loc]
    #
    #     if nominal_action == [1, 0]:
    #         idx = np.random.choice(3, size=1, p=[0.9, 0.05, 0.05])[0]
    #         actual_actions_list = [[1, 0], [pi / 2, 1], [pi / 2, -1]]
    #         actual_action = actual_actions_list[idx]
    #     else:
    #         actual_action = nominal_action
    #
    #     # assume perfect control
    #     # actual_action = nominal_action
    #     print('step', i, 'loc', cur_loc,
    #           'cur loc utility', planner.utility[cur_loc],
    #           'action', actual_action)
    #     i += 1
    #     robot.publish_discrete_control_one(actual_action)
    #
    # assert robot.get_current_discrete_state()[:2] == planner._get_goal_position(), "Didn't reach the goal."

    ## print utility table

    ####################### <- end of Executing

    # TODO: FILL ME!
    # After you implement your planner, send the plan/policy generated by your
    # planner using appropriate execution calls.
    # e.g., robot.publish_discrete_control(planner.action_seq, goal)

    # save your action sequence
    # result = np.array(planner.action_seq)
    # np.savetxt("DSPA_map1_{}_{}.txt".format(goal[0], goal[1]), result, fmt="%.2e")


    # for MDP, please dump your policy table into a json file
    dump_action_table(planner.action_table, "DSPA_com1building_{}_{}.json".format(goal[0], goal[1]))
    print('action table', planner.action_table)