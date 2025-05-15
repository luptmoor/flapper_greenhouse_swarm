
from Simulation import Simulation, run, SwarmAggregatorLSTM
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


# if __name__ == '__main__':


#     # bt = BehaviourTree(random_tree=True)
#     # bt.save2file()
#     start_time = time.perf_counter();

#     for s in range(20):
       
#         sim = Simulation(seed=s);
#         run(sim);
        
#     end_time = time.perf_counter();
#     print(f"Sim time: {end_time - start_time} s")

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
    model = SwarmAggregatorLSTM()
    load_flat_params(model, params)
    model.eval()
    sim = Simulation(vis)
    fitness = run(sim, model, vis, gen)  # Must return a scalar
    return -fitness  # CMA-ES minimizes


def save_cma_state(es, filename="cma_state.npz"):
    state = {
        'xmean': es.mean,
        'sigma': es.sigma,
        'generation': es.countiter,
        'population_size': es.popsize,
        'best': es.best.get()[0],
        'best_fitness': es.best.get()[1]
    }
    np.savez(filename, **state)
    print(f"[✔] CMA-ES state saved to {filename}")

# --- Load CMA-ES state and reinitialize optimizer ---
def load_cma_state(filename="cma_state.npz"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No CMA-ES state file found at {filename}")
    
    data = np.load(filename)
    xmean = data['xmean']
    sigma = float(data['sigma'])
    popsize = int(data['population_size'])

    es = cma.CMAEvolutionStrategy(xmean, sigma, {'popsize': popsize})
    print(f"[✔] CMA-ES state loaded from {filename}")
    return es

# --- Save current population to CSV for inspection ---
def save_cma_params_csv(population, filename="population.csv"):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        for individual in population:
            writer.writerow(individual)
    print(f"[✔] Population saved to {filename}")

# --- Load population from CSV (optional utility) ---
def load_cma_params_csv(filename="population.csv"):
    population = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            population.append([float(val) for val in row])
    return population



def save_fitnesses(fitnesses, filename="fitnesses.csv"):
    """
    append current fitness list to a csv file
    """
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fitnesses)
    print(f"[✔] Fitnesses saved to {filename}")
            




if __name__ == '__main__':


    start_time = time.perf_counter();
    np.random.seed(0)
    torch.manual_seed(0)


    model = SwarmAggregatorLSTM()
    x0 = flatten_params(model)
    sigma0 = 0.3

    #es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': 25})
    es = cma.CMAEvolutionStrategy.load("cma_state.npz")
    gen = 1;
    vis = False
    while not es.stop():
        print(f'generation {gen}')
        
        if gen % 168 == 0: vis = True
        else: vis = False
        print(f'visuals: {vis}')

        solutions = es.ask()
        fitnesses = [fitness_fn(s, vis, gen) for s in solutions]
        save_fitnesses(fitnesses)

        es.tell(solutions, fitnesses)
        es.logger.add()
        es.disp()

        es.save("cma_state.npz")
        gen += 1;

    # ---- Save best ----
    best_params = es.result.xbest
    save_cma_params_csv(best_params)


    end_time = time.perf_counter();
    print(f"Sim time: {end_time - start_time} s")


    
    # Train attraction-repulsion swarming in obstacle-free environment to learn behaviour for exploration/exploitation tradeoff
    # Train tofnet without fruits to learn collision avoidance  OR use serban's RL solution


