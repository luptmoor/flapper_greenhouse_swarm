
import copy
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
import glob

    
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
    #     #bt.load_from_file('gen_44/bt_0_score_0.113.json')
    #     #bt.load_from_file('genX_12/bt_0_score_0.137.json') # regularly incrases altitude up to explore only the top of the greenhouse
    #     #bt.load_from_file('genX_1/bt_3_score_0.155.json') # stays close to the ground
    #     #bt.load_from_file('genX_5/bt_1_score_0.130.json') # ascends if encounters obstacles, only turns when risen to ceiling
    #     #bt.load_from_file('genX_6/bt_0_score_0.131.json') # very similar
    #     #bt.load_from_file('genX_20/bt_1_score_0.121.json') # also 
    #     #bt.load_from_file('genX_81/bt_0_score_0.085_mod.json') # stays close to the ground, turns always right, manages to explore other rows sometimes
       
    #     #bt.load_from_file('genC_155/bt_9_score_0.055.json') 
    #     bt.load_from_file('gen_64/bt_1_score_0.054.json')

        
    #     score = run(sim, bt, gen=12312, phenotype=0)







    # json_folder = 'genX_1'  # Change this to your folder path
    # population = []
    # for json_file in glob.glob(os.path.join(json_folder, '*.json')):
    #     bt = BehaviourTree()
    #     bt.load_from_file(json_file)
    #     population.append(bt)

    

    # # # Read population and scores from pickle
    # # with open(f'genX_{1}/population.pkl', 'rb') as f:
    # #     data = pickle.load(f)
    # #     population = data['population']
    # #     score_list = data['scores']


    folder_prefix = "genC3"

    population = [BehaviourTree(seed=i) for i in range(20, 20 + POPULATION_SIZE)]  # Initialize population with random BTs


    for gen in range(1, 301):
        print()
        print()
        print(f"Generation {gen}")

        score_list = []


        #  1. Simulate
        for i in range(len(population)):
            scores = np.zeros(N_SIMRUNS)
            for k in range(N_SIMRUNS):
                sim = Simulation(vis=VISUALISE, seed=k)
                scores[k] = run(sim, population[i], gen=gen, vis=VISUALISE, phenotype=i)

            score_list.append(np.mean(scores))
            population[i].save_to_json(f'{folder_prefix}_{gen}/bt_{i}_score_{score_list[-1]:.3f}.json')  
            population[i].save_to_pdf(f'{folder_prefix}_{gen}/bt_{i}_score_{score_list[-1]:.3f}')
        

        #  2.  Save the population and scores using pickle
        with open(f'{folder_prefix}_{gen}/population.pkl', 'wb') as f:
            pickle.dump({'population': population, 'scores': score_list}, f)
        
        
        
        population, score_list = zip(*sorted(zip(population, score_list), key=lambda x: x[1], reverse=True))


        # #  3a. Absolute selection
        # selection = [copy.deepcopy(bt) for bt in population[:POPULATION_SIZE//2]]
        # population = selection + [copy.deepcopy(bt) for bt in selection] + [BehaviourTree(seed=np.random.randint(0, 1000))]


        #  3b. Tournament selection

        # Start with elite
        new_population = [copy.deepcopy(population[i]) for i in range(N_ELITE)]

        while len(new_population) < POPULATION_SIZE:
            tournament_scores = []
            indices = []
            for i in range(TOURNAMENT_SIZE):
                index = np.random.randint(0, len(score_list))
                while index in indices:
                    index = np.random.randint(0, len(score_list))
                indices.append(index)
                tournament_scores.append(score_list[index])

            winner_index = indices[np.argmax(tournament_scores)]
            new_population.append(copy.deepcopy(population[winner_index]))

       
        #  4. Crossover
        mates = copy.deepcopy(new_population)
        for i in range (N_ELITE, int(round((POPULATION_SIZE - N_ELITE) * CROSSOVER_RATE))):
            new_population[i].root.crossover(random.choice(mates).root)


        #  5. Mutate
        for i in range(N_ELITE + int(round((POPULATION_SIZE - N_ELITE) * CROSSOVER_RATE)), POPULATION_SIZE):
            if random.uniform(0, 1) < P_MICROMUTATION:
                new_population[i].root.micromutate()
            else:
                new_population[i].root.macromutate()



        population = new_population

        
      






       
