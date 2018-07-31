import numpy as np
import agent
import robot

AGENT_ELEMENT = ["MOBILE_BASE", "DISTANCE_SENSOR"]
ENV_ELEMENT = []

MOTOR_SPEED = 1
RANGE_OBSTACLES = 0.5
RANGE_CHANGE = 0.08
RANGE_DAMAGE = 0.05
STEP_TIME = 1

INPUT_VARIABLES = {
    "laser_left": np.linspace(0, RANGE_OBSTACLES, 4),
    "laser_left_front": np.linspace(0, RANGE_OBSTACLES, 4),
    "laser_front_left": np.linspace(0, RANGE_OBSTACLES, 4),
    "laser_front": np.linspace(0, RANGE_OBSTACLES, 4),
    "laser_front_right": np.linspace(0, RANGE_OBSTACLES, 4),
    "laser_right_front": np.linspace(0, RANGE_OBSTACLES, 4),
    "laser_right": np.linspace(0, RANGE_OBSTACLES, 4)
}
OUTPUT_VARIABLES = {
    "left_wheel": np.linspace(-MOTOR_SPEED, MOTOR_SPEED, 3),
    "right_wheel": np.linspace(-MOTOR_SPEED, MOTOR_SPEED, 3)
}
INITIAL_STATE = 0  # (usually overwritten by the fist observation)
INITIAL_POLICY = 0
REWARDS = np.array([-1, -0.3,-0.01, 1.0])

n_inputs = -1
in_values = [None]
in_names = [None]
in_sizes = [int]
n_states = -1
in_data = [None]

n_outputs = -1
out_values = [None]
out_names = [None]
out_sizes = [int]
n_actions = -1
out_data = [None]

def execute_action(act_factors):
    left, right = act_factors[0], act_factors[1]
    if left < 0 and right < 0:
        left =right = MOTOR_SPEED * 2
    elif (left == 0 and right < 0) or (left < 0 and right == 0):
        left = right = 0
    robot.move_wheels(left, right)
    return


def get_reward():
    distance1 = robot.sensor["laser_left"]
    distance2 = robot.sensor["laser_left_front"]
    distance3 = robot.sensor["laser_front_left"]
    distance4 = robot.sensor["laser_front"]
    distance6 = robot.sensor["laser_front_right"]
    distance7 = robot.sensor["laser_right_front"]
    distance8 = robot.sensor["laser_right"]

    change = robot.mobilebase_change
    n_collisions=(int(distance1 < RANGE_DAMAGE)+int(distance2<RANGE_DAMAGE)+int(distance3<RANGE_DAMAGE)+
                  int(distance4<RANGE_DAMAGE)+int(distance6<RANGE_DAMAGE)+
                  int(distance7<RANGE_DAMAGE)+int(distance8<RANGE_DAMAGE))
    r=REWARDS[2]
    if n_collisions >2:
        r=REWARDS[0]
    elif n_collisions <=2 and n_collisions >=1:
        r= REWARDS[1]
    elif  change > RANGE_CHANGE:
        r = REWARDS[3]
    return r

def get_input_data():
    global in_data
    for i,item in enumerate(in_names):
        #print('item:',item)
        in_data[i]=robot.sensor[item]
    return in_data

def setup():
    agent.setup_task()

