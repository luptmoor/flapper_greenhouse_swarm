from numba import njit
from settings import *
import numpy as np
from geometry_fns import pyramid_intersects_cuboid, plot_poly, cuboid_points_from_params, pyramid_intersects_bounds
import matplotlib.pyplot as plt


_half_extent = R_TOF * np.tan(TOF_HFOV / 2)
_apex = np.array([0.0, 0.0, 0.0])
_base1 = np.array([R_TOF, -_half_extent, -_half_extent])
_base2 = np.array([R_TOF,  _half_extent, -_half_extent])
_base3 = np.array([R_TOF, -_half_extent,  _half_extent])
_base4 = np.array([R_TOF,  _half_extent,  _half_extent])




###################### ACTION FUNCTIONS ########################


def apf_avoidance(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    Function to calculate the avoidance vector for an array of N drones using artificial potential fields.
    Assume that minimum avoidance distance is not met when this fn is called
    PARAMETERS:
     - K_REP: Repulsive gain
     - R_REP: Repulsive distance threshold
    return: vx_cmd, vz_cmd, r_cmd, msg, status
    """

    if np.sum(active_array) < 2: return {}, 'success', 'avd'

    vx_cmd = 0.0
    vy_cmd = 0.0
    vz_cmd = 0.0
    r_cmd = 0.0

    # Calculate repulsive force from other drones
    for i in range(N_DRONES - 1):
        if not active_array[i]:
            continue
        dx = swarm_array[3 * i]
        dy = swarm_array[3 * i + 1]
        dz = swarm_array[3 * i + 2]
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if dist > params["AVD_R_REP"]:
            continue

        dist = max(dist, 0.001)  # Avoid division by zero
        
        # Repulsive force (inverse distance)
        fx = params["AVD_K_REP"] * dx / (dist ** 2)
        fy = params["AVD_K_REP"] * dy / (dist ** 2)
        fz = params["AVD_K_REP"] * dz / (dist ** 2)    
        
        # Forward command is along the force direction
        vx_cmd += fx
        vy_cmd += fy
        vz_cmd += fz

    if np.sqrt(vx_cmd ** 2 + vy_cmd ** 2 + vz_cmd ** 2) < 0.001:
        return {}, 'success', 'avd'

    target_heading = np.arctan2(vx_cmd, vy_cmd)
    heading_error = target_heading - heading
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    #print(f'target heading {target_heading*57.3:.2f} deg, heading error {heading_error*57.3:.2f} deg')

    vz_cmd = np.clip(vz_cmd, -V_UP_MAX, V_UP_MAX)
    vx_cmd = np.clip(np.sqrt(vx_cmd **2 + vy_cmd **2), 0, V_FORWARD_MAX)  # Forward speed is the magnitude of the force vector
    r_cmd = np.clip(heading_error * 0.5, -YAWRATE_MAX, YAWRATE_MAX)  # Proportional control for yaw rate
    
    if np.abs(heading_error) > params["AVD_HEADING_PRECISION"]:
        vx_cmd = 0.0

    return {"vx": vx_cmd, "vz": vz_cmd, "r": r_cmd}, 'running', 'avd'



     
def approach(params, x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    approach a fruit until it is considered observed
    """

    return  {}, 'running', 'app'


def follow_wall(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    stay to the wall and move up and down along it
    """

    return  {}, 'running', 'wall'


def random_walk(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    random turn and climb commands with constant forward speed
    """

    return {"vx": params["EXP_VX"], "vz": np.random.uniform(-params["EXP_VZ_SPREAD"], params["EXP_VZ_SPREAD"]), "r": np.random.uniform(-params["EXP_R_SPREAD"], params["EXP_R_SPREAD"])}, 'running', 'exp'


def disperse(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    
    """
    move away from the other drones
    """

    if np.sum(active_array) < 2: return {}, 'success', 'disp'

    x_avg = 0.0
    y_avg = 0.0
    for i in range(N_DRONES - 1):
        if abs(swarm_array[3*i]) < 0.001 and abs(swarm_array[3*i+1]) < 0.001 and abs(swarm_array[3*i+2]) < 0.001:
            continue

        x_avg += swarm_array[3*i] - x
        y_avg += -swarm_array[3*i+1] + y

    x_avg /= (np.sum(active_array) - 1)
    x_avg = -x_avg
    y_avg /= (np.sum(active_array) - 1)


    #print(f"Dispersing from average position ({x_avg:.2f}, {y_avg:.2f}) from {x:.2f}, {y:.2f}")
    dx = x - x_avg
    dy = y - y_avg
    target_heading = np.arctan2(dy, dx)# + np.pi 

    #print(f"Target heading: {target_heading * 57.3:.2f} deg")

    if target_heading > np.pi:
        target_heading -= 2 * np.pi
    if target_heading < -np.pi:
        target_heading += 2 * np.pi


    heading_error = target_heading - heading

    #print(f"Heading error: {heading_error * 57.3:.2f} deg")

    if np.abs(heading_error) < params["DISP_HEADING_PRECISION"]: vx_cmd = params["DISP_VX"]
    else: vx_cmd = 0

    r_cmd = heading_error * params["DISP_K_HEADING"]

    r_cmd = np.clip(r_cmd, -YAWRATE_MAX, YAWRATE_MAX)


    return {"vx": vx_cmd, "r": r_cmd}, 'running', 'disp'
    


def turn_right(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    brake and rotate right until the path is clear
    """

    return {"vx": 0.0, "vz": 0.0, "r": params["RGHT_TURN_RATE"]}, 'running', 'right'



def turn_left(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    brake and rotate left until the path is clear
    """

    return {"vx": 0.0, "vz": 0.0, "r": -params["LEFT_TURN_RATE"]}, 'running', 'left'


def ascend(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    ascend until the ceiling is reached
    """

    if z < CEILING - 0.1:
        return {"vx": 0.0, "vz": params["ASC_VZ"], "r": 0.0}, 'running', 'asc'
    else:
        return {}, 'success', 'asc'
    

def descend(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    descend until the ground is reached
    """

    if z > 0.1:
        return {"vx": 0.0, "vz": -params["DESC_VZ"], "r": 0.0}, 'running', 'desc'
    else:
        return {}, 'success', 'desc'
    
    
def brake(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    brake until the speed is 0
    """

    if np.sqrt(vx**2 + vz**2) > 0.05:
        return {"vx": 0.0, "vz": 0.0, "r": 0.0}, 'running', 'brake'
    else:
        return {"vx": 0.0, "vz": 0.0, "r": 0.0}, 'success', 'brake'



def send_message(params, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    send a message to the other drones
    """

    return  {"msg": 1}, 'success', 'msg'



################### CONDITION FUNCTIONS #######################

def fruit_counter(params, tick, x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if the fruit counter is greater than 0
    """

    return 'failure', False


def discovery_rate(params, tick, x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if the discovery rate is greater than 0
    """

    return 'failure', False


def fruit_visible(params, tick,  x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    check if the fruit is visible
    """

    return 'failure', False


def swarm_spread(params, tick, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    check if the swarm is spread out
    """

    if np.sum(active_array) < 2: return 'success', False

    d_list = [1000]
    for i in range(N_DRONES - 1):
        if abs(swarm_array[3*i]) < 0.001 and abs(swarm_array[3*i+1]) < 0.001 and abs(swarm_array[3*i+2]) < 0.001:
            continue

        d = np.sqrt(swarm_array[3*i]**2 + swarm_array[3*i+1]**2 + swarm_array[3*i+2]**2) 
        d_list.append(d)
    
    d_list.sort()
    index = min(max(np.sum(active_array) // 3, 0), len(d_list) - 1)

    if d_list[index] < params["SPRD_THRESHOLD"]:
        return 'failure', False
    else:
        return 'success', False



def path_clear(params, tick, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    njit-compatible: check if any cuboid corner is inside the FOV pyramid
    """

    # Rotation matrix for heading (around Z)
    c = np.cos(heading)
    s = np.sin(heading)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    t = np.array([x, y, z])

    # Transform pyramid vertices to world frame
    apex_w = R @ _apex + t
    base1_w = R @ _base1 + t
    base2_w = R @ _base2 + t
    base3_w = R @ _base3 + t
    base4_w = R @ _base4 + t

    fov_pyramid = np.array([apex_w, base1_w, base2_w, base3_w, base4_w])
    for i in range(obstacle_array.shape[0]):
        #cuboid_points = cuboid_points_from_params(obstacle_array[i])
        
        if pyramid_intersects_cuboid(fov_pyramid, obstacle_array[i]) or pyramid_intersects_bounds(fov_pyramid):
            # print(f"Obstacle {i} intersects with FOV pyramid")
            # print(f'cuboid points: {cuboid_points}')
            # # Plot
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # plot_poly(ax, cuboid_points, color='blue', alpha=0.4)
            # plot_poly(ax, fov_pyramid, color='green', alpha=0.6)
            # ax.scatter(*cuboid_points.T, color='blue')
            # ax.scatter(*fov_pyramid.T, color='green')
            # ax.set_xlim(0, WIDTH)
            # ax.set_ylim(0, HEIGHT)
            # ax.set_zlim(0, CEILING)
            # # axis labels
            # ax.set_xlabel('X (m)')
            # ax.set_ylabel('Y (m)')
            # ax.set_zlabel('Z (m)')
            # plt.show()
            # dummy = input("Press Enter to continue...")

            return 'failure', False
        
    #print("Path is clear")
    return 'success', False


def min_distance(params, tick, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    check if the minimum distance to other drones is greater than 1.0m
    """

    d_list = [10000]
    for i in range(N_DRONES - 1):
        if abs(swarm_array[3*i]) < 0.001 and abs(swarm_array[3*i+1]) < 0.001 and abs(swarm_array[3*i+2]) < 0.001:
            continue

        d = np.sqrt(swarm_array[3*i]**2 + swarm_array[3*i+1]**2 + swarm_array[3*i+2]**2) 
        d_list.append(d)

    if min(d_list) > params["MINP_DISTANCE"]:
        return 'success', False
    else: return 'failure', False
    

def message_received(params, tick, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    check if a message was received
    """
    if np.sum(msg_array) > 0: return 'success', False
    else: return 'failure', False


def random_condition(params, tick, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    check if a random number is greater than 0.5
    """


    randnr = np.random.uniform(0, 1)
    #print(f"Random condition: {randnr}")
    if randnr > params["RND_THRESHOLD"]: return 'success', False
    else: return 'failure', False


def timer_condition(params, tick, x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    check if the timer is greater than X seconds
    """

    if tick > params["TIMER_THRESHOLD"]:
        print(f"Timer condition met at tick {tick}")
        return 'success', True
    else:
        return 'failure', False


    
   

############ Lists #############


action_strings = [
    #'Approach',
    'Avoid other drones',
    'Turn right',
    'Turn left',
    #'Follow wall',
    'Random Walk',
    'Disperse',
    'Send message',
    'Ascend',
    'Descend',
    'Brake',
]

actions = {
#   "Approach":  approach,
    "Avoid other drones":  apf_avoidance,
    "Turn right":  turn_right,
    "Turn left":  turn_left,
#   "Follow wall":  follow_wall,
    "Random Walk":  random_walk,
    "Disperse":  disperse,
    "Send message":  send_message,
    "Ascend": ascend,
    "Descend": descend,
    "Brake": brake,
}

condition_strings = [
    #'Fruit visible?',
    #'# discovered fruit > X ?',
    #'# new fruit last 30s < X ?',
    'Path clear?',
    'Minimum peer distance > X ?',
    'Message received?',
    'Random > X ?',
    'Timer > X ?',
]

conditions = {
#   "Fruit visible?": fruit_visible,
#   "# discovered fruit > X ?": fruit_counter,
#   "# new fruit last 30s < X ?: discovery_rate,
    "Path clear?": path_clear,
    "Minimum peer distance > X ?": min_distance,
    "Message received?": message_received,
    "Random > X ?": random_condition,
    "Swarm spread out?": swarm_spread,
    "Timer > X ?": timer_condition
}

frequencies = {
    "Path clear?": 10,
    "Minimum peer distance > X ?": 10,
    "Message received?": 20,
    "Random > X ?": 100,
    "Swarm spread out?": 10,
    "Timer > X ?": 10
}


param_dicts = {
    "Avoid other drones": {"AVD_K_REP": 1.0, "AVD_R_REP": 1.0, "AVD_HEADING_PRECISION": 0.17},  # Heading precision in radians
    "Turn right": {"RGHT_TURN_RATE": 15 / 57.3},  # 15 degrees in radians
    "Turn left": {"LEFT_TURN_RATE": 15 / 57.3},  # 15 degrees in radians
    "Random Walk": {"EXP_VX": 0.3, "EXP_VZ_SPREAD": 0.3, "EXP_R_SPREAD": 15 / 57.3},
    "Disperse": {"DISP_K_HEADING": 0.5, "DISP_VX": 0.3, "DISP_HEADING_PRECISION": 10 / 57.3},
    "Send message": {},
    "Ascend": {"ASC_VZ": 0.3},
    "Descend": {"DESC_VZ": 0.3},
    "Brake": {},
    "Path clear?": {},
    "Minimum peer distance > X ?": {"MINP_DISTANCE": 1.0},
    "Message received?": {},
    "Random > X ?": {"RND_THRESHOLD": 0.5},
    "Swarm spread out?": {"SPRD_THRESHOLD": 2.0},
    "Timer > X ?": {"TIMER_THRESHOLD": 100}
}

param_ranges = {
    "AVD_K_REP": (0.1, 10.0),
    "AVD_R_REP": (0.5, 10.0),
    "AVD_HEADING_PRECISION": (0.01, 45.0 / 57.3),  # radians
    "RGHT_TURN_RATE": (0.1, YAWRATE_MAX),  
    "LEFT_TURN_RATE": (0.1, YAWRATE_MAX), 
    "EXP_VX": (0.1, V_FORWARD_MAX),
    "EXP_VZ_SPREAD": (0.1, V_UP_MAX),
    "EXP_R_SPREAD": (0.1, YAWRATE_MAX),  
    "DISP_K_HEADING": (0.1, 5.0),
    "DISP_VX": (0.1, V_FORWARD_MAX),
    "DISP_HEADING_PRECISION": (0.01, 45.0 / 57.3),  # 45 degrees in radians
    "ASC_VZ": (0.0, V_UP_MAX),
    "DESC_VZ": (0.0, V_UP_MAX),
    "MINP_DISTANCE": (0.0, 8.0),     # m
    "RND_THRESHOLD": (0.0, 1.0),
    "SPRD_THRESHOLD": (0.0, 8.0),    # m
    "TIMER_THRESHOLD": (50, 1500)  # seconds
}