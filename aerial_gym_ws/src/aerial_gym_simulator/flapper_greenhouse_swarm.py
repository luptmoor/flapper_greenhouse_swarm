
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
from behaviour_tree import BehaviourTree
from bt_evolution import BTEvolution
import torch
from datetime import datetime
time = datetime.now().strftime("%d_%m_%H_%M")

MAX_TIME            =       600 * 100       # steps
POPULATION_SIZE     =        50
N_GENERATIONS       =        50
BASEPATH            =       "simresults/results_"





if __name__ == "__main__":

    # Parse arguments from CLI
    args = get_args()


    
    # Build simulation
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="greenhouse_env",
        robot_name="flapper",
        controller_name="lee_position_control",
        args=None,
        device="cuda:0",
        num_envs=8,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    evolution = BTEvolution(env_manager, population_size=POPULATION_SIZE, n_generations=N_GENERATIONS, tmax=MAX_TIME, filepath=BASEPATH + time)

    bt = BehaviourTree(3)

    blackboard = {
        "current_time": 0,
        "absolute_x": 0,
        "absolute_y": 0,
        "absolute_z": 0,
        "heading": 0,
        "visit_list": [],
        "tof_array": None,
        "peer_distances": [],
        "fruit_visible": False,
    }


    evolution.simulate_bt(bt, MAX_TIME)

    