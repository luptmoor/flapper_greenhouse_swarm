
from Simulation import Simulation
import cma
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from settings import *
import os
import pandas as pd
import random
from behaviour_tree import BehaviourTree


if __name__ == '__main__':

    # params = 13 * [random.uniform(-1, 1)]

    # bt = BehaviourTree(random_tree=True)
    # bt.save2file()
    for i in range(20):
        bt = BehaviourTree('random.json', random_tree=True, seed=i);
        sim = Simulation(bt=bt);
        sim.run();

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