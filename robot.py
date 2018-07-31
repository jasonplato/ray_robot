import math
import numpy as np
import rl_vrep

mobilebase_pose = np.full(3, -1, dtype=np.float64)
last_mobilebase_pose = np.full(3, -1, dtype=np.float64)
mobilebase_change = 0

obstacle_dist = np.full(rl_vrep.N_LASERS, -1, dtype=np.float64)

AGENT_ELEMENTS = []
ENV_ELEMENTS = []

initiated = False

sensor = {}


def setup(AGENT_ELEM, ENV_ELEM):
    global AGENT_ELEMENTS, ENV_ELEMENTS
    global mobilebase_pose, last_mobilebase_pose, mobilebase_change
    global obstacle_dist, initiated, sensor

    AGENT_ELEMENTS = AGENT_ELEM
    ENV_ELEMENTS = ENV_ELEM

    mobilebase_pose = np.full(3, -1, dtype=np.float64)
    last_mobilebase_pose = np.full(3, -1, dtype=np.float64)
    mobilebase_change = 0

    obstacle_dist = np.full(rl_vrep.N_LASERS, -1, dtype=np.float64)

    update()
    update()

    return


def update():
    """ update robot & environment state (sensors, locations...) """
    global mobilebase_pose, last_mobilebase_pose, mobilebase_change
    global obstacle_dist, sensor

    if "DISTANCE_SENSOR" in AGENT_ELEMENTS:
        obstacle_dist = get_obstacle_distance()
    if "MOBILE_BASE" in AGENT_ELEMENTS:
        last_mobilebase_pose = mobilebase_pose
        mobilebase_pose = get_mobilebase_pose()
        mobilebase_change = distance(mobilebase_pose, last_mobilebase_pose)

    sensor["mobile_x"] = mobilebase_pose[0]
    sensor["mobile_y"] = mobilebase_pose[1]
    sensor["mobile_theta"] = mobilebase_pose[2]

    sensor["laser_left"] = obstacle_dist[0]
    sensor["laser_left_front"] = obstacle_dist[1]
    sensor["laser_front_left"] = obstacle_dist[2]
    sensor["laser_front"] = obstacle_dist[3]
    sensor["laser_front_right"] = obstacle_dist[4]
    sensor["laser_right_front"] = obstacle_dist[5]
    sensor["laser_right"] = obstacle_dist[6]
    return


def move_wheels(left, right):
    rl_vrep.move_wheels(left, right)
    return


def stop_wheels():
    rl_vrep.stop_wheels()
    return


def get_obstacle_distance():
    d = rl_vrep.get_distance()
    return d


def get_mobilebase_pose():
    p = rl_vrep.get_pose()
    return p


def distance(pose1, pose2):
    change = math.sqrt((pose1[0] - pose2[0]) ** 2 + (pose1[1] - pose2[1]) ** 2)
    return change


def start():
    rl_vrep.start()
    return


def stop():
    rl_vrep.stop()
    return


def setup_vrep():
    rl_vrep.setup()
    return


def connect():
    rl_vrep.connect()


def disconnect():
    rl_vrep.disconnect()
    return
