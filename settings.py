import numpy as np

# Environment   scale: 1 px = 1cm
SCALE                =    100                                      # px / m
WIDTH                =     15                                       # m
HEIGHT               =     8                                       # m
BORDERSIZE           =    10                                       # px, cm
LAUNCHPAD_FRAC       =    0.2                                      # -, used to determine drones' launchpad size
CEILING              =    3.0                                      # m, maximum height of drones

# Trees
N_TREES_PER_ROW      =   30                                        # -
R_TREE_MIN           =   0.20                                      # m
R_TREE_MAX           =   0.40                                      # m
R_TREE_AVG           =   (R_TREE_MAX + R_TREE_MIN) / 2             # px, mean for normal distribution
R_TREE_STD           =   (R_TREE_MAX - R_TREE_MIN) / 6             # px, standard deviation for normal distribution
TREE_HEIGHT          =    1.2                                      # m, physical quantity
FRUIT_PROB           =    0.5
FRUIT_MIN_HEIGHT     =    0.2                                      # m
R_FRUIT              =    0.05                                     # m


# Static Drone Parameters
N_DRONES             =    5                                        # -, number of drones
R_DRONE              =    0.14                                        # cm drone radius
V_DRONE_MAX          =  500                                        # px / s,  5 m / s
A_DRONE_MAX          =  330                                        # px / s^2, 3.3 m / s^2
CAMERA_VFOV          =  30 / 57.3                                   # rad
CAMERA_HFOV          =  40 / 57.3                                   # rad

# Simulation parameters
SEED                 =   44
DT                   =    0.05                                       # s, timestep per tick
T_MAX                =  120                                         # s, 1min, maximum simulation duration
VISUALISE            =  True                                        # Boolean deciding if simulation shall be visualised
REALTIME             =  False                                       # Boolean deciding if visuals shall be real-time
VIEW                 =    0                                         # Variable for different views in visualisation
SENSITIVITY_ANALYSIS =  False                                        # Boolean deciding if noise should be added to certain parameters
NOISE                =    0.0                                     # -, Noise amplitude if in sensitivity analysis
MANUAL               = False



BASEPATH            =       "simresults/results_"

## Evolution
POPULATION_SIZE     =        100
N_GENERATIONS       =        150
P_MICROMUTATION     =         0.2
P_MACROMUTATION     =         0.1
ELITISM_RATE        =         0.04
N_ELITE             =       int(ELITISM_RATE * POPULATION_SIZE)
N_TOURNAMENTS       =        10


## Behaviour Trees
BT_SEED             =        49
BT_MAX_CHILDREN     =         6
BT_MAX_DEPTH        =         6
P_BT_COMPOSITE      =         0.25
P_BT_CONDITION      =         0.25
P_BT_ACTION         =         0.25
P_BT_SEQUENCE       =         0.5


## Action and reading limits
YAWRATE_MAX         =        np.pi/2        # rad/s
V_FORWARD_MAX       =         2             # m/s
V_BACKWARD_MAX      =         0.5           # m/s
V_LEFT_MAX          =         0.5           # m/s
V_RIGHT_MAX         =         0.5           # m/s
V_UP_MAX            =         0.5           # m/s
V_DOWN_MAX          =         0.5           # m/slee_velocity_control
MIN_BAT_THRESHOLD   =        10             # s
MAX_BAT_THRESHOLD   =       600             # s

    
# Actions
ACTION_VARS = ['r', 'vx', 'vz', 'message', 'memory', 'tofnet', 'swarmnet']
ACTION_LIMITS = {
    'r': (-YAWRATE_MAX, YAWRATE_MAX),    
    'vx':  (-V_BACKWARD_MAX, V_FORWARD_MAX),
    'vz':  (-V_DOWN_MAX,     V_LEFT_MAX),
    'message':  (-1.0, 1.0),
    'memory':   (-1.0, 1.0)
}

# Conditions
READING_VARS = ['fruit_visible', 'elapsed_battery_time', 'memory']
READING_LIMITS = {
    'elapsed_battery_time': (MIN_BAT_THRESHOLD, MAX_BAT_THRESHOLD),
    'memory':   (-1.0, 1.0),
}





# RGB colours for visualisation
GREEN                = (50, 150, 50)
BROWN                = (117, 60, 26)
GREY                 = (150, 150, 150)
RED                  = (180, 0, 0)
YELLOW               = (255, 255, 0)
ORANGE               = (255, 134, 0)
PINK                 = (255, 0, 255)
BLUE                 = (50, 50, 180)
WHITE                = (255, 255, 255)
BLACK                = (0, 0, 0)

BEETLE_COLOURS = {'idle': RED, 'land': ORANGE, 'tree': YELLOW, 'escape': PINK}
TYPE_COLOURS = {'drone': GREY, 'tree': BROWN}


# CMA-ES settings
RUNS_PER_SOLUTION     =    3                                        # -, for how many conditions each solution is tested
N_GENERATIONS         =   90                                        # -, number of generations in CMA-ES
N_POP                 =   20                                        # -, number of genotypes per generation

# Parameter 0
MIN_R_VIS_TREE        =   10
MAX_R_VIS_TREE        =  150
MU_R_VIS_TREE         = (MAX_R_VIS_TREE + MIN_R_VIS_TREE) / 2
RANGE_R_VIS_TREE      = MAX_R_VIS_TREE - MIN_R_VIS_TREE

# Parameter 1
MIN_K_TREE            =    0
MAX_K_TREE            =  150
MU_K_TREE             = (MAX_K_TREE + MIN_K_TREE) / 2
RANGE_K_TREE          = MAX_K_TREE - MIN_K_TREE

# Parameter 2
MIN_R_VIS_BEETLE         =   10
MAX_R_VIS_BEETLE         =  200
MU_R_VIS_BEETLE          = (MAX_R_VIS_BEETLE + MIN_R_VIS_BEETLE) / 2
RANGE_R_VIS_BEETLE       = MAX_R_VIS_BEETLE - MIN_R_VIS_BEETLE

# Parameter 3
MIN_K_BEETLE             =   -150
MAX_K_BEETLE             =    0
MU_K_BEETLE              = (MAX_K_BEETLE + MIN_K_BEETLE) / 2
RANGE_K_BEETLE           = MAX_K_BEETLE - MIN_K_BEETLE

# Parameter 4
MIN_R_VIS_NEARDRONE   =   10
MAX_R_VIS_NEARDRONE   =  150
MU_R_VIS_NEARDRONE    = (MAX_R_VIS_NEARDRONE + MIN_R_VIS_NEARDRONE) / 2
RANGE_R_VIS_NEARDRONE = MAX_R_VIS_NEARDRONE - MIN_R_VIS_NEARDRONE

# Parameter 5
MIN_K_NEARDRONE       =    0
MAX_K_NEARDRONE       =  150
MU_K_NEARDRONE        = (MAX_K_NEARDRONE + MIN_K_NEARDRONE) / 2
RANGE_K_NEARDRONE     = MAX_K_NEARDRONE - MIN_K_NEARDRONE

# Parameter 6
MIN_R_VIS_FARDRONE    =  MAX_K_NEARDRONE
MAX_R_VIS_FARDRONE    =  500
MU_R_VIS_FARDRONE     = (MAX_R_VIS_FARDRONE + MIN_R_VIS_FARDRONE) / 2
RANGE_R_VIS_FARDRONE  = MAX_R_VIS_FARDRONE - MIN_R_VIS_FARDRONE

# Parameter 7
MIN_K_FARDRONE        =   -0.5
MAX_K_FARDRONE        =    0.5
MU_K_FARDRONE         = (MAX_K_FARDRONE + MIN_K_FARDRONE) / 2
RANGE_K_FARDRONE      = MAX_K_FARDRONE - MIN_K_FARDRONE

# Parameter 8
MIN_R_ACTIVITY        =  MIN_R_VIS_FARDRONE
MAX_R_ACTIVITY        =  MAX_R_VIS_FARDRONE
MU_R_ACTIVITY         = (MAX_R_ACTIVITY + MIN_R_ACTIVITY) / 2
RANGE_R_ACTIVITY      = MAX_R_ACTIVITY - MIN_R_ACTIVITY

# Parameter 9
MIN_K_ACTIVITY        =   -0.5
MAX_K_ACTIVITY        =    0.5
MU_K_ACTIVITY         = (MAX_K_ACTIVITY + MIN_K_ACTIVITY) / 2
RANGE_K_ACTIVITY      = MAX_K_ACTIVITY - MIN_K_ACTIVITY

# Parameter 10
MIN_V_MIN             =    0
MAX_V_MIN             =   20
MU_V_MIN              = (MAX_V_MIN + MIN_V_MIN) / 2
RANGE_V_MIN           = MAX_V_MIN - MIN_V_MIN

# Parameter 11
MIN_V_MAX             =   MAX_V_MIN
MAX_V_MAX             =  V_DRONE_MAX
MU_V_MAX              = (MAX_V_MAX + MIN_V_MAX) / 2
RANGE_V_MAX           = MAX_V_MAX - MIN_V_MAX

# Parameter 12
MIN_C                 =   0
MAX_C                 =   1
MU_C                  = (MAX_C + MIN_C) / 2
RANGE_C               = MAX_C - MIN_C


def noise(k):
    """
    if sensitivity analysis setting is enabled, then a value in the interval [1-k, 1+k] is returned, otherwise 1
    :param k: noise amplitude, should lie in [0, 1]
    :return: noise factor
    """
    if SENSITIVITY_ANALYSIS:
        randnr = np.random.random() * 2 * k  # random number between 0 and 2k
        return 1 + randnr - k
    else:
        return 1
