from numba import njit

###################### ACTION FUNCTIONS ########################


def apf_avoidance(bb):
    """
    Function to calculate the avoidance vector for an array of N drones using artificial potential fields.
    Assume that minimum avoidance distance is not met when this fn is called
    return: vx_cmd, vz_cmd, r_cmd
    """
    return  0, 0, 0, 'running'


     
def approach(bb):
    """
    approach a fruit until it is considered observed
    """

    return  0, 0, 0, 'running'


def follow_wall(bb):
    """
    stay to the wall and move up and down along it
    """

    return  0, 0, 0, 'running'


def random_walk(bb):
    """
    random turn and climb commands with constant forward speed
    """

    return  0, 0, 0, 'running'


def disperse(bb):
    """
    move away from the other drones
    """

    return  0, 0, 0, 'running'


def clear_path(bb):
    """
    brake and rotate left until the path is clear
    """

    return  0, 0, 0, 'running'




################### CONDITION FUNCTIONS #######################

def fruit_counter(bb):
    """
    check if the fruit counter is greater than 0
    """

    return 'failure'


def discovery_rate(bb):
    """
    check if the discovery rate is greater than 0
    """

    return 'failure'


def fruit_visible(bb):
    """
    check if the fruit is visible
    """

    return 'failure'


def path_clear(bb):
    """
    check if the path is clear
    """

    return 'failure'


def min_distance(bb):
    """
    check if the minimum distance to other drones is greater than 0
    """

    return 'failure'
