
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

import matplotlib.image as mpimg
from graphviz import Digraph
import tempfile

    
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
            












def visualize_bt_live(bt_generator):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ion()
    img_obj = None

    for bt in bt_generator:
        # Create and render Graphviz image
        dot = bt.plot()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            dot.render(tmpfile.name, format='png', cleanup=True)
            img_path = tmpfile.name + '.png'
            img = mpimg.imread(img_path)

        # Update Matplotlib window
        ax.clear()
        ax.imshow(img)
        ax.axis('off')
        fig.canvas.draw()
        plt.pause(0.1)

        os.remove(img_path)  # Clean up image

    plt.ioff()
    plt.show()



def simulate_bt_steps():
    """
    Function to test realtime behaviour tree visualization with a random selection of BTs"""
    for i in range(100):
        bt = BehaviourTree(seed=i)

        blackboard = {
            'fruit_visible': np.random.randint(0, 2),
            'elapsed_battery_time': np.random.uniform(0, 600),
            'memory': np.random.uniform(-1, 1),
        }
        print(blackboard)
        bt.feed_forward(blackboard)
        yield bt
        dummy = input("Press Enter to continue...")





if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0) 

    # for s in range(10):
    #     sim = Simulation(vis=True, seed=s)
    #     bt = BehaviourTree()
    #     bt.load_from_file('manual_v4.json')
    #     score = run(sim, bt, gen=0, phenotype=0)



    population = [BehaviourTree(seed=i) for i in range(21, 30)]
    
    for gen in range(1, 100):
        print()
        print()
        print(f"Generation {gen}")

        score_list = []

        #  1. Simulate
        for i in range(len(population)):
            sim = Simulation(vis=VISUALISE, seed=gen)
            score = run(sim, population[i], gen=gen, phenotype=i)
            score_list.append(score)
            population[i].save_to_json(f'gen_{gen}/bt_{i}_score_{score:.3f}.json')  
            population[i].save_to_pdf(f'gen_{gen}/bt_{i}_score_{score:.3f}')
        

        #  2.  Save the population and scores using pickle
        with open(f'gen_{gen}/population.pkl', 'wb') as f:
            pickle.dump({'population': population, 'scores': score_list}, f)
        

        #  3. Sort population by descending score (highest to lowest)
        population, score_list = zip(*sorted(zip(population, score_list), key=lambda x: x[1], reverse=True))

        
        selection = population[:5]
        print(f"Best score: {score_list[0]}")
        population = selection + selection


        #  4. Mutate
        for i in range(2, len(population)):
            population[i].root.macromutate()

        for i in range(2, len(population)):
            population[i].root.micromutate()

        
      






        # # Read population and scores from pickle
        # with open(f'gen_{gen}/population.pkl', 'rb') as f:
        #     data = pickle.load(f)
        #     population = data['population']
        #     score_list = data['scores']

