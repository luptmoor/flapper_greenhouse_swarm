
from Simulation import Simulation, run
import cma
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from settings import *
import os
import pandas as pd
import random
from behaviour_tree import BehaviourTree
import time
import torch


if __name__ == '__main__':


    # bt = BehaviourTree(random_tree=True)
    # bt.save2file()
    start_time = time.perf_counter();

    for s in range(20):
       
        sim = Simulation(seed=s);
        run(sim);
        
    end_time = time.perf_counter();
    print(f"Sim time: {end_time - start_time} s")

    # bt.set_path('mutated.json')
    # bt.root.macromutate();
    # bt.show();
    # sim = Simulation(bt=bt);
    # sim.run()
    
    # centroid of fruit visited in polar body coords
    # aggregate, dispersea
    # first explore to identify fruit locations, then exploit centroid
    # Train attraction-repulsion swarming in obstacle-free environment to learn behaviour for exploration/exploitation tradeoff
    # Train tofnet without fruits to learn collision avoidance  OR use serban's RL solution