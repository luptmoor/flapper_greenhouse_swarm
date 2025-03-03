MAX_TIME            =       600 * 100       # steps
BASEPATH            =       "simresults/results_"

## Evolution
POPULATION_SIZE     =        50
N_GENERATIONS       =        50
P_MICROMUTATION     =         0.2
P_MACROMUTATION     =         0.1


## Behaviour Trees
BT_SEED             =        42
BT_MAX_CHILDREN     =         6
BT_MAX_DEPTH        =         6
P_BT_COMPOSITE      =         0.25
P_BT_CONDITION      =         0.25
P_BT_ACTION         =         0.25
P_BT_SEQUENCE       =         0.5


## Action and reading limits
YAWRATE_MAX         =        90             # deg/s
V_FORWARD_MAX       =         2             # m/s
V_BACKWARD_MAX      =         0.5           # m/s
V_LEFT_MAX          =         0.5           # m/s
V_RIGHT_MAX         =         0.5           # m/s
V_UP_MAX            =         0.5           # m/s
V_DOWN_MAX          =         0.5           # m/slee_velocity_control
MIN_BAT_THRESHOLD   =        10             # s
MAX_BAT_THRESHOLD   =       600             # s

    
# Actions
ACTION_VARS = ['r', 'vx', 'vy', 'vz', 'message', 'memory', 'tofnet', 'swarmnet']
ACTION_LIMITS = {
    'r': (-YAWRATE_MAX, YAWRATE_MAX),    
    'vx':  (-V_BACKWARD_MAX, V_FORWARD_MAX),
    'vy':  (-V_RIGHT_MAX,    V_LEFT_MAX),
    'vz':  (-V_DOWN_MAX,     V_LEFT_MAX),
    'message':  (-1.0, 1.0),
    'memory':   (-1.0, 1.0),
}

# Conditions
READING_VARS = ['fruit_visible', 'elapsed_battery_time', 'memory']
READING_LIMITS = {
    'elapsed_battery_time': (MIN_BAT_THRESHOLD, MAX_BAT_THRESHOLD),
    'memory':   (-1.0, 1.0),
}