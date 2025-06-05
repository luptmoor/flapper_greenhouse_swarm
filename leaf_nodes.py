from numba import njit
from settings import *
import numpy as np

###################### ACTION FUNCTIONS ########################


def apf_avoidance(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    Function to calculate the avoidance vector for an array of N drones using artificial potential fields.
    Assume that minimum avoidance distance is not met when this fn is called
    return: vx_cmd, vz_cmd, r_cmd
    """
    return  0, 0, 0, 0, 'running'


     
def approach(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    approach a fruit until it is considered observed
    """

    return  0, 0, 0, 0, 'running'


def follow_wall(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    stay to the wall and move up and down along it
    """

    return  0, 0, 0, 0, 'running'


def random_walk(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    random turn and climb commands with constant forward speed
    """

    return  0.3, np.random.uniform(-0.3, 0.3), np.random.uniform(-10/57.3, 10/57.3), 0, 'running'


def disperse(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    
    """
    move away from the other drones
    """

    x_avg = np.sum([swarm_array[3*i] + x for i in range(N_DRONES - 1) if np.abs(swarm_array[3*i] + x) > 0.01]) / np.sum(active_array)
    y_avg = np.sum([swarm_array[3*i+1] + y for i in range(N_DRONES - 1) if np.abs(swarm_array[3*i+1] + y) > 0.01]) / np.sum(active_array)

    dy = x - x_avg
    dx = y - y_avg
    target_heading = np.arctan2(dy, dx) + np.pi 

    if target_heading > np.pi:
        target_heading -= 2 * np.pi
    if target_heading < -np.pi:
        target_heading += 2 * np.pi


    heading_error = target_heading - heading

    if heading_error < 10/57.3: vx_cmd = 0.3
    else: vx_cmd = 0

    r_cmd = heading_error * 0.5

    return  vx_cmd, 0, r_cmd, 0, 'running'


def clear_path(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    brake and rotate left until the path is clear
    """

    return  0, 0, 10/57.3, 0, 'running'


def send_message(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    send a message to the other drones
    """

    return  0, 0, 0, 1, 'success'



################### CONDITION FUNCTIONS #######################

def fruit_counter(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if the fruit counter is greater than 0
    """

    return 'failure'


def discovery_rate(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if the discovery rate is greater than 0
    """

    return 'failure'


def fruit_visible(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if the fruit is visible
    """

    return 'failure'


def path_clear(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if the path is clear
    """

    return 'failure'


def min_distance(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if the minimum distance to other drones is smaller than 0.5m
    """

    d = [np.sqrt(swarm_array[i]**2 + swarm_array[i+1]**2 + swarm_array[i+2]**2) for i in range(N_DRONES-1)]

    if min(d) < 0.5:
        return 'success'
    else: return 'failure'
    

def message_received(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if a message was received
    """

    if np.sum(msg_array) > 0: return 'success'
    else: return 'failure'


def random_condition(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if a random number is greater than 0.5
    """

    if np.random.uniform(0, 1) > 0.5: return 'success'
    else: return 'failure'



############ Lists #############


action_strings = [
    'Approach',
    'Avoid other drones',
    'Turn right',
    'Follow wall',
    'Random Walk',
    'Disperse',
    'Send message',
]

actions = [
    approach,
    apf_avoidance,
    clear_path,
    follow_wall,
    random_walk,
    disperse,
    send_message
]

condition_strings = [
    'Fruit visible?',
    '# discovered fruit > X ?',
    '# new fruit last 30s < X ?',
    'Path clear?',
    'Minimum peer distance < X ?',
    'Message received?',
    'Random > 0.5 ?',
]

conditions = [
    fruit_visible,
    fruit_counter,
    discovery_rate,
    path_clear,
    min_distance,
    message_received,
    random_condition
]