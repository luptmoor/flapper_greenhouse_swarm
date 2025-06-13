from numba import njit
from settings import *
import numpy as np
from geometry_fns import pyramid_intersects_cuboid, plot_poly, cuboid_points_from_params, pyramid_intersects_bounds
import matplotlib.pyplot as plt
# import trimesh
# from trimesh.transformations import translation_matrix, rotation_matrix


# _half_extent = R_TOF * np.tan(TOF_HFOV / 2)

# # Apex at origin
# _apex = np.array([0, 0, 0])

# # Square base lies at distance R_TOF along +X
# _base1 = np.array([R_TOF, -_half_extent, -_half_extent])
# _base2 = np.array([R_TOF,  _half_extent, -_half_extent])
# _base3 = np.array([R_TOF, -_half_extent,  _half_extent])
# _base4 = np.array([R_TOF,  _half_extent,  _half_extent])


# # Combine vertices
# _vertices = np.array([
#     _base1,
#     _base2,
#     _base3,
#     _base4,
#     _apex
#    # np.array([0, 0, 0])  # Center of the base for side triangles
# ])

# # Faces: base (2 triangles) + 4 side triangles
# _faces = [
#     [0, 1, 3],  # base triangle 1
#     [0, 3, 2],  # base triangle 2
#     [4, 0, 1],  # side 1
#     [4, 1, 3],  # side 2
#     [4, 3, 2],  # side 3
#     [4, 2, 0],  # side 4
# ]

# _FOV_pyramid = trimesh.Trimesh(vertices=_vertices, faces=_faces, process=False)
# _FOV_pyramid.fix_normals()  # Ensure normals are correct



_half_extent = R_TOF * np.tan(TOF_HFOV / 2)
_apex = np.array([0.0, 0.0, 0.0])
_base1 = np.array([R_TOF, -_half_extent, -_half_extent])
_base2 = np.array([R_TOF,  _half_extent, -_half_extent])
_base3 = np.array([R_TOF, -_half_extent,  _half_extent])
_base4 = np.array([R_TOF,  _half_extent,  _half_extent])




###################### ACTION FUNCTIONS ########################


def apf_avoidance(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    Function to calculate the avoidance vector for an array of N drones using artificial potential fields.
    Assume that minimum avoidance distance is not met when this fn is called
    return: vx_cmd, vz_cmd, r_cmd, msg, status
    """

    if np.sum(active_array) < 2: return {}, 'success', 'avd'

    # Parameters for APF
    K_REP = 1.0  # Repulsive gain

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

        if dist > 1.0:
            continue

        dist = max(dist, 0.001)  # Avoid division by zero
        
        # Repulsive force (inverse distance)
        fx = K_REP * dx / (dist ** 2)
        fy = K_REP * dy / (dist ** 2)
        fz = K_REP * dz / (dist ** 2)
        # Calculate target heading for the repulsive force
        

        
        # Forward command is along the force direction
        vx_cmd += fx
        vy_cmd += fy
        vz_cmd += fz



    target_heading = np.arctan2(vx_cmd, vy_cmd)
    heading_error = target_heading - heading
    heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi

    #print(f'target heading {target_heading*57.3:.2f} deg, heading error {heading_error*57.3:.2f} deg')

    vz_cmd = np.clip(vz_cmd, -V_UP_MAX, V_UP_MAX)
    vx_cmd = np.clip(np.sqrt(vx_cmd **2 + vy_cmd **2), 0, V_FORWARD_MAX)  # Forward speed is the magnitude of the force vector
    r_cmd = np.clip(heading_error * 0.5, -YAWRATE_MAX, YAWRATE_MAX)  # Proportional control for yaw rate
    
    if np.abs(heading_error) > 10/57.3:
        vx_cmd = 0.0

    return {"vx": vx_cmd, "vz": vz_cmd, "r": r_cmd}, 'running', 'avd'



     
def approach(x, y, z, heading, vx, vz, r, swarm_array, fruit_x, fruit_y, fruit_z, fruit_side_array, fruit_disc_array, obstacle_array, active_array, msg_array):
    """
    approach a fruit until it is considered observed
    """

    return  {}, 'running', 'app'


def follow_wall(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    stay to the wall and move up and down along it
    """

    return  {}, 'running', 'wall'


def random_walk(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    random turn and climb commands with constant forward speed
    """

    return {"vx": 0.3, "vz": np.random.uniform(-0.3, 0.3), "r": np.random.uniform(-15/57.3, 15/57.3)}, 'running', 'exp'


def disperse(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    
    """
    move away from the other drones
    """

    print('disperse')
    if np.sum(active_array) < 2: return {}, 'success', 'disp'

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

    return {"vx": vx_cmd, "vz": 0.0, "r": r_cmd}, 'running', 'disp'
    


def turn_right(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    brake and rotate right until the path is clear
    """

    return {"vx": 0.0, "vz": 0.0, "r": YAWRATE_MAX}, 'running', 'right'



def turn_left(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    brake and rotate left until the path is clear
    """

    return {"vx": 0.0, "vz": 0.0, "r": -YAWRATE_MAX}, 'running', 'left'



def send_message(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    send a message to the other drones
    """

    return  {"msg": 1}, 'success', 'msg'



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


def swarm_spread(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):

    if np.sum(active_array) < 2: return 'success'

    d_list = [1000]
    for i in range(N_DRONES - 1):
        if abs(swarm_array[3*i]) < 0.001 and abs(swarm_array[3*i+1]) < 0.001 and abs(swarm_array[3*i+2]) < 0.001:
            continue

        d = np.sqrt(swarm_array[3*i]**2 + swarm_array[3*i+1]**2 + swarm_array[3*i+2]**2) 
        d_list.append(d)
    
    d_list.sort()
    index = min(max(np.sum(active_array) // 3, 0), len(d_list) - 1)

    if d_list[index] < 2.0:
        return 'failure'
    else:
        return 'success'



def path_clear(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
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

            return 'failure'
        
    #print("Path is clear")
    return 'success'


def min_distance(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    check if the minimum distance to other drones is greater than 1.0m
    """
    d_list = [10000]
    for i in range(N_DRONES - 1):
        if abs(swarm_array[3*i] < 0.001) and abs(swarm_array[3*i+1]) < 0.001 and abs(swarm_array[3*i+2]) < 0.001:
            continue

        d = np.sqrt(swarm_array[3*i]**2 + swarm_array[3*i+1]**2 + swarm_array[3*i+2]**2) 
        d_list.append(d)

    if min(d_list) > 1.0:
        return 'success'
    else: return 'failure'
    

def message_received(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    check if a message was received
    """
    if np.sum(msg_array) > 0: return 'success'
    else: return 'failure'


def random_condition(x, y, z, heading, vx, vz, r, swarm_array, obstacle_array, active_array, msg_array):
    """
    check if a random number is greater than 0.5
    """
    randnr = np.random.uniform(0, 1)
    #print(f"Random condition: {randnr}")
    if randnr > 0.5: return 'success'
    else: return 'failure'



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
]

actions = {
#   "Approach":  approach,
    "Avoid other drones":  apf_avoidance,
    "Turn right":  turn_right,
    "Turn left":  turn_left,
#   "Follow wall":  follow_wall,
    "Random Walk":  random_walk,
    "Disperse":  disperse,
    "Send message":  send_message
}

condition_strings = [
    #'Fruit visible?',
    #'# discovered fruit > X ?',
    #'# new fruit last 30s < X ?',
    'Path clear?',
    'Minimum peer distance > X ?',
    'Message received?',
    'Random > 0.5 ?',
]

conditions = {
#   "Fruit visible?": fruit_visible,
#   "# discovered fruit > X ?": fruit_counter,
#   "# new fruit last 30s < X ?: discovery_rate,
    "Path clear?": path_clear,

    "Minimum peer distance > X ?": min_distance,
    "Message received?": message_received,
    "Random > 0.5 ?": random_condition,
    "Swarm spread out?": swarm_spread
}

frequencies = {
    "Path clear?": 10,
    "Minimum peer distance > X ?": 10,
    "Message received?": 20,
    "Random > 0.5 ?": 100,
    "Swarm spread out?": 10
}