
from Simulation import Simulation
import cma
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from settings import *
import os
import pandas as pd
import random


if __name__ == '__main__':
    print('Hello World')

    params = 13 * [random.uniform(-1, 1)]

    sim = Simulation(params=params);
    sim.run();