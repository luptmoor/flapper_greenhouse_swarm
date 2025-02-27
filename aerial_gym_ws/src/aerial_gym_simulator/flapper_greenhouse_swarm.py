
from aerial_gym.sim.sim_builder import SimBuilder
from aerial_gym.utils.helpers import get_args
from behaviour_tree import BehaviourTree
from bt_evolution import BTEvolution
import torch
from datetime import datetime
from settings import *
time = datetime.now().strftime("%d_%m_%H_%M")





if __name__ == "__main__":

    # Parse arguments from CLI
    args = get_args()


    
    # Build simulation
    env_manager = SimBuilder().build_env(
        sim_name="base_sim",
        env_name="greenhouse_env",
        robot_name="flapper",
        controller_name="lee_velocity_control",
        args=None,
        device="cuda:0",
        num_envs=5,
        headless=args.headless,
        use_warp=args.use_warp,
    )

    evolution = BTEvolution(env_manager, population_size=POPULATION_SIZE, n_generations=N_GENERATIONS, tmax=MAX_TIME, filepath=BASEPATH + time)

    #bt = BehaviourTree(3)

   

    blackboard = {
        "elapsed_battery_time": 0,
        "absolute_x": 0,
        "absolute_y": 0,
        "absolute_z": 0,
        "heading": 0,
        "visit_list": 0,
        "tof": 0,
        "peer_distances": [],
        "fruit_visible": False,
        'swarmneural': 0,
        'memory': 0.2,
    }

    bt = BehaviourTree(random_tree=True)
    bt.save2file('test_tree.json')

    #bt.feed_forward(blackboard=blackboard)

    evolution.simulate_bt(bt, MAX_TIME)

    