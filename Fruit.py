import matplotlib.pyplot as plt

from Entity import *
from settings import *
import numpy as np





class Fruit(Entity):
    def __init__(self, name, x, y, z):
        """
        Simulated Bug
        :param name:
        :param x:
        :param y:
        :param r_col:
        :param speed:
        :param heading:
        :param r_vis:
        :param mode:
        """
        super().__init__(name, 'fruit', x, y, R_FRUIT)
        self.tree = None
        self.z = z
        self.identified = False

        self.counter = 0
        self.record = [0.0]
    

    def advance(self):
        """
        performs the integration of the beetle's behaviour according to its current mode
        :param dt: timestep size, is variable because beetles perform half the step size twice
        :return: None
        """

        self.counter += 1

        if self.counter % 10 == 0:
            self.record.append(self.record[-1] + 10 * DT)

        
    def reset_counter(self):
        self.identified = True;
        self.record[-1] = 0.0;