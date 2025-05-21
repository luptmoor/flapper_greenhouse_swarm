from numba import njit

###################### ACTION FUNCTIONS ########################


def apf_avoidance(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    Function to calculate the avoidance vector for an array of N drones using artificial potential fields.
    Assume that minimum avoidance distance is not met when this fn is called
    return: vx_cmd, vz_cmd, r_cmd
    """
    return  0, 0, 0, 'running'


     
def approach(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    approach a fruit until it is considered observed
    """

    return  0, 0, 0, 'running'


def follow_wall(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    stay to the wall and move up and down along it
    """

    return  0, 0, 0, 'running'


def random_walk(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    random turn and climb commands with constant forward speed
    """

    return  0, 0, 0, 'running'


def disperse(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    move away from the other drones
    """

    return  0, 0, 0, 'running'


def clear_path(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    brake and rotate left until the path is clear
    """

    return  0, 0, 0, 'running'




################### CONDITION FUNCTIONS #######################

def fruit_counter(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    check if the fruit counter is greater than 0
    """

    return 'failure'


def discovery_rate(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    check if the discovery rate is greater than 0
    """

    return 'failure'


def fruit_visible(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    check if the fruit is visible
    """

    return 'failure'


def path_clear(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    check if the path is clear
    """

    return 'failure'


def min_distance(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array, obstacle_array):
    """
    check if the minimum distance to other drones is greater than 0
    """

    return 'failure'




############ Lists #############


action_strings = [
    'Approach',
    'Avoid other drones',
    'Turn left',
    'Follow wall',
    'Random Walk',
    'Disperse',
]

actions = [
    approach,
    apf_avoidance,
    clear_path,
    follow_wall,
    random_walk,
    disperse
]

condition_strings = [
    'Fruit visible?',
    '# discovered fruit > X ?',
    '# new fruit last 30s < X ?',
    'Path clear?',
    'Minimum peer distance < X ?'
]

conditions = [
    fruit_visible,
    fruit_counter,
    discovery_rate,
    path_clear,
    min_distance
]