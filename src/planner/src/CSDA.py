#!/usr/bin/env python

# repo token: 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
# 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
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

######### -> start of Newly added
from Queue import PriorityQueue


######### <- end of Newly added

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

            if self.collision_checker(x, y):
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
            tuple -- next x, y, theta; return None if has collision
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

    def publish_continuous_control(self, action_seq, goal):
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
        plan = [ContinuousAction(action[0], action[1]) for action in action_seq]
        proxy(plan)
        assert self.is_close_to_goal(goal), "Didn't reach the goal."

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


class CSDAPlanner(Planner):
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
        self.path_seq = None # output
        self.aug_map = None  # occupancy grid with inflation
        self.action_table = {}

        self.world_width = world_width
        self.world_height = world_height
        self.resolution = world_resolution
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
                # inflation circle whose radius = inflation ratio
                # if euclidean_distance_to_center(x, y) <= self.inflation_ratio:
                #     nei_relative_position.append([x, y])
                # inflation square whose side lenght = 2 * inflation ratio
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
        ###################################<- end of FILL ME

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

        #### for CSDA hybrid A*
        # hybrid A* algorithm implementation, assuming continuous states
        ## Euclidean distance as the heuristic H
        def h_euclidean(x1, y1):
            x2, y2 = self._get_goal_position()
            return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        def pose_is_close_to_goal(x, y):
            return math.sqrt((x - self._get_goal_position()[0]) ** 2
                             + (y - self._get_goal_position()[1]) ** 2) < 0.1

        grid_resolution = 0.1

        def loc_to_grid_index(x, y):
            return int(x / grid_resolution), int(y / grid_resolution)



        init_x, init_y, init_theta = init_pose
        g_score = {loc_to_grid_index(init_x, init_y): 0}
        f_score = {loc_to_grid_index(init_x, init_y): 0 + h_euclidean(init_x, init_y)}

        frontier = PriorityQueue()  # f score, priority queue: location (x, y, theta)
        frontier.put((f_score[loc_to_grid_index(init_x, init_y)], (init_x, init_y, init_theta)))

        path_parent = dict()
        path_control_from_parent = dict()

        current_x, current_y, current_theta = 1, 1, 0  # dummy initialization

        while not frontier.empty():
            current_f, (current_x, current_y, current_theta) = frontier.get()

            if pose_is_close_to_goal(current_x, current_y):
                break

            for _ in range(10):
                # sample a neighboring node that can be reached within one timestep
                # uniform sample v and w
                v = np.random.uniform(0.5, 1)
                w = np.random.uniform(-pi, pi)

                # print('current_x, current_y, current_theta, v, w',current_x, current_y, current_theta, v, w)
                nei_pose = self.motion_predict(current_x, current_y, current_theta, v, w)
                if nei_pose is None:
                    continue

                nei_x, nei_y, nei_theta = nei_pose
                # make sure next x, y is within boundary and occupancy rate is below 100
                # print('nei_x, nei_y, nei_theta, collision', nei_x, nei_y, nei_theta, self.collision_checker(nei_x, nei_y))

                if not (0 <= nei_x < self.world_width
                        and 0 <= nei_y < self.world_height):
                    continue
                if self.collision_checker(nei_x, nei_y):
                    continue

                cur_x_grid_index, cur_y_grid_index = loc_to_grid_index(current_x, current_y)
                nei_x_grid_index, nei_y_grid_index = loc_to_grid_index(nei_x, nei_y)
                tmp_nei_g_score = g_score[(cur_x_grid_index, cur_y_grid_index)] + abs(
                    v / w * (nei_theta - current_theta))
                tmp_nei_f_score = tmp_nei_g_score + h_euclidean(nei_x, nei_y)

                # print(0)


                if (nei_x_grid_index, nei_y_grid_index) not in f_score \
                        or tmp_nei_f_score < f_score[(nei_x_grid_index, nei_y_grid_index)]:
                    # print(1)
                    g_score[(nei_x_grid_index, nei_y_grid_index)] = tmp_nei_g_score
                    f_score[(nei_x_grid_index, nei_y_grid_index)] = tmp_nei_f_score
                    frontier.put((tmp_nei_f_score, (nei_x, nei_y, nei_theta)))
                    path_parent[(nei_x, nei_y, nei_theta)] = (current_x, current_y, current_theta)
                    path_control_from_parent[(nei_x, nei_y, nei_theta)] = (v, w)

            # if i < 2:
            #     i += 1
            #     print('i current_f, (current_x, current_y, current_theta)', current_f, (current_x, current_y, current_theta))
            #     print('priority queue', frontier.queue)

        # print('path_parent', path_parent)
        # print('path_control_from_parent', path_control_from_parent)

        # get action sequence according to sequence path tree
        self.action_seq = []
        self.path_seq = [(current_x, current_y, current_theta)]
        while (current_x, current_y) != (init_x, init_y):
            self.action_seq.append(path_control_from_parent[(current_x, current_y, current_theta)])
            (current_x, current_y, current_theta) = path_parent[(current_x, current_y, current_theta)]
            self.path_seq.append((current_x, current_y, current_theta))

        self.action_seq.reverse()
        self.path_seq.reverse()
        ### for DSPA
        ########### save action table for DSPA

    def xy_to_1d_grid_index(self, x, y):
        return y * self.world_width + x

    def unit_to_world(self, x_unit, y_unit):
        return int(x_unit / self.resolution), int(y_unit / self.resolution)

    def collision_checker_wrt_original_map(self, x, y):
        return (0 <= x < self.world_width and 0 <= y < self.world_height) \
               and self.map[self.xy_to_1d_grid_index(x, y)] == 100

    def collision_checker(self, x, y):
        """TODO: FILL ME!
        unit collision checker
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
        x_world, y_world = self.unit_to_world(x, y)
        # print('x_world, y_world', x_world, y_world)
        # print('xy_to_1d_grid_index', self.xy_to_1d_grid_index(int(x_world), int(y_world)))
        # print('aug_map', self.aug_map[self.xy_to_1d_grid_index(int(x_world), int(y_world))])
        return (0 <= x_world < self.world_width and 0 <= y_world < self.world_height) \
               and self.aug_map[self.xy_to_1d_grid_index(int(x_world), int(y_world))] == 100

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

    ############# choose different planner
    # planner = DSDAPlanner(width, height, resolution, inflation_ratio=inflation_ratio)
    # planner.set_goal(goal[0], goal[1])
    # if planner.goal is not None:
    #     planner.generate_plan(robot.get_current_discrete_state())
    # print('action sequence', planner.action_seq)
    # robot.publish_discrete_control(planner.action_seq, goal)
    #############

    planner = CSDAPlanner(width, height, resolution, inflation_ratio=16)

    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan(robot.get_current_continuous_state())
    print('action sequence', planner.action_seq)
    print('path sequence', planner.path_seq)
    robot.publish_continuous_control(planner.action_seq, goal)

    # publish each action one by one
    # for i, action in enumerate(planner.action_seq):
    #     print('step', i)
    #     print('actual path', robot.get_current_continuous_state())
    #     print('planned path', planner.path_seq[i])
    #     robot.publish_continuous_control_one(action)

    # an ought to be collided point
    # print('1.69, 2', planner.collision_checker(1.69, 2))
    # print('1.82, 1.6', planner.collision_checker(1.82, 1.6))


    # check collision points around (2, 2)
    # print('inflation_ratio', planner.inflation_ratio)
    # for x in range(10,30):
    #     for y in range(10,30):
    #         print('here', float(x)/10, float(y)/10, planner.collision_checker(float(x)/10, float(y)/10))

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
    # mock_action_plan = [(0, -1), (0, -1), (1, 0), (1, 0), (0, -1), (0, 1), (0, -1), (1, 0), (1, 0)]

    # robot.publish_discrete_control(planner.action_seq, goal)
    # TODO: FILL ME!
    # After you implement your planner, send the plan/policy generated by your
    # planner using appropriate execution calls.
    # e.g., robot.publish_discrete_control(planner.action_seq, goal)

    # save your action sequence
    result = np.array(planner.action_seq)
    np.savetxt("CSDA_com1building_{}_{}.txt".format(goal[0], goal[1]), result, fmt="%.2e")

    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')
