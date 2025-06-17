import numpy as np

# Environment   scale: 1 px = 1cm
SCALE                =    100                                      # px / m
WIDTH                =     15                                       # m
HEIGHT               =     8                                       # m
BORDERSIZE           =    10                                       # px, cm
LAUNCHPAD_FRAC       =    0.1                                      # -, used to determine drones' launchpad size
CEILING              =    3.0                                      # m, maximum height of drones
SQUEEZE_RANGE      =      0.5                                      # m, distance from ground and ceiling in which vertical velocity is gradually limited

# Trees
N_TREES_PER_ROW      =   30                                        # -
R_TREE_MIN           =   0.20                                      # m
R_TREE_MAX           =   0.40                                      # m
R_TREE_AVG           =   (R_TREE_MAX + R_TREE_MIN) / 2             # px, mean for normal distribution
R_TREE_STD           =   (R_TREE_MAX - R_TREE_MIN) / 6             # px, standard deviation for normal distribution
TREE_HEIGHT          =    1.8                                      # m, physical quantity
FRUIT_PROB           =    0.5
FRUIT_MIN_HEIGHT     =    0.2                                      # m
R_FRUIT              =    0.05                                     # m
N_FRUIT              =   30


# Static Drone Parameters
N_DRONES             =    5                                        # -, number of drones
R_DRONE              =    0.14                                        # cm drone radius
TOF_VFOV             =  23 / 57.3                                   # rad
TOF_HFOV             =  23 / 57.3                                   # rad
R_TOF                =   1.0


# First order lag drone model
TAU_VX               =   0.3
TAU_VZ               =   0.5
TAU_R                =   0.1

# Simulation parameters
SEED                 =   44
DT                   =    0.05                                       # s, timestep per tick
T_MAX                =  200                                         # s, 1min, maximum simulation duration
MAX_TICKS            = int(T_MAX // DT)
VISUALISE            =  True                                       # Boolean deciding if simulation shall be visualised
SHOW_BT              =  True                                       # Boolean deciding if behaviour trees shall be visualised   
REALTIME_BT          =  False                                       # Boolean deciding if behaviour trees shall be visualised in real-time      
REALTIME             =  False                                     # Boolean deciding if visuals shall be real-time
VIEW                 =    0                                         # Variable for different views in visualisation
SENSITIVITY_ANALYSIS =  False                                        # Boolean deciding if noise should be added to certain parameters
NOISE                =    0.0                                     # -, Noise amplitude if in sensitivity analysis
MANUAL               = False
VOXEL_SIZE           =  0.5                                        # m, size of voxel grid cells


BASEPATH            =       "simresults/results_"

## Evolution
POPULATION_SIZE     =        100
N_GENERATIONS       =        150
P_MICROMUTATION     =         0.5
P_MACROMUTATION     =         0.2
ELITISM_RATE        =         0.04
N_ELITE             =       int(ELITISM_RATE * POPULATION_SIZE)
N_TOURNAMENTS       =        10


## Behaviour Trees
BT_SEED             =        49
BT_MAX_CHILDREN     =         4
BT_MAX_DEPTH        =         4
P_BT_COMPOSITE      =         0.25
P_BT_CONDITION      =         0.25
P_BT_ACTION         =         0.25
P_BT_SEQUENCE       =         0.5


## Action and reading limits
YAWRATE_MAX         =        np.pi/2        # rad/s
V_FORWARD_MAX       =         1.0           # m/s
V_BACKWARD_MAX      =         0.0           # m/s
V_LEFT_MAX          =         0.0           # m/s
V_RIGHT_MAX         =         0.0           # m/s
V_UP_MAX            =         0.5           # m/s
V_DOWN_MAX          =         0.5           # m/s
MIN_BAT_THRESHOLD   =        10             # s
MAX_BAT_THRESHOLD   =       600             # s






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
