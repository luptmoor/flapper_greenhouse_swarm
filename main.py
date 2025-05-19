
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


    model = WeightedDeepSet()
    x0 = flatten_params(model)
    sigma0 = 0.3

    es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': 25})

    # OR
    # with open("cma_state.pkl", "rb") as f:
    #     es = pickle.load(f)

    gen = 1;
    vis = False
    while not es.stop():
        print(f'generation {gen}')
        
        if gen % 40 == 0: vis = True
        else: vis = False
        print(f'visuals: {vis}')

        solutions = es.ask()
        fitnesses = [fitness_fn(s, vis, gen) for s in solutions]
        save_fitnesses(gen, es.sigma, fitnesses, "fitnesses_weighteddeepset.csv")

        es.tell(solutions, fitnesses)
        es.logger.add()
        es.disp()

        with open("cma_state_weighteddeepset.pkl", "wb") as f:
            pickle.dump(es, f)

        gen += 1;

    # ---- Save best ----
    best_params = es.result.xbest

    
    # Train attraction-repulsion swarming in obstacle-free environment to learn behaviour for exploration/exploitation tradeoff
    # Train tofnet without fruits to learn collision avoidance  OR use serban's RL solution


