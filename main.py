
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

    params = 13 * [random.uniform(-1, 1)]

    bt = BehaviourTree()
    bt.save2file()

    sim = Simulation(bt=bt);
    sim.run();