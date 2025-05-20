
from Simulation import Simulation, run, WeightedDeepSet
import cma
import csv
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
import pickle



    
    # centroid of fruit visited in polar body coords
    # aggregate, dispersea
    # first explore to identify fruit locations, then exploit centroid
    # Train attraction-repulsion swarming in obstacle-free environment to learn behaviour for exploration/exploitation tradeoff
    # Train tofnet without fruits to learn collision avoidance  OR use serban's RL solution



    # ---- Helper functions ----
def flatten_params(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy()

def load_flat_params(model, flat_params):
    i = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(torch.tensor(flat_params[i:i+numel]).view(p.shape))
        i += numel


def fitness_fn(params, vis, gen):
    model = WeightedDeepSet()
    load_flat_params(model, params)
    model.eval()
    sim = Simulation(vis)
    fitness = run(sim, model, vis, gen)  # Must return a scalar
    return -fitness  # CMA-ES minimizes




def save_fitnesses(gen, sigma, fitnesses, filename="fitnesses.csv"):
    """
    append current fitness list to a csv file
    """
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([gen, sigma, -np.mean(fitnesses), -np.min(fitnesses), *fitnesses])
    #print(f"[✔] Fitnesses saved to {filename}")
            




if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)

    tree = BehaviourTree()
    tree.save_to_json('test.json')
    tree.load_from_file('test.json')

    tree.save_to_pdf('test_tree.pdf')
    #tree.save()

