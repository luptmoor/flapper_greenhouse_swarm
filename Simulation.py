import numpy as np
from Entity import Entity
from Drone import Drone
from Fruit import Fruit
from Visuals import Visuals
from settings import *
import random
import torch



def check_collision(entity1, entity2, margin=0):
    """
    checks if entities 1 and 2 have intersecting pixels.
    :param entity1: Entity 1
    :param entity2: Entity 2
    :param margin: (float) minimum distance between the two entities to return False (no collision)
    :return: (boolean) True if Entities collide, False if Entities do not collide.
    """
    if entity1 is None or entity2 is None:
        return False

    if entity1.name == entity2.name:
        return False

    if np.sqrt((entity1.x - entity2.x) ** 2 + (entity1.y - entity2.y) ** 2) <= entity1.r_col + entity2.r_col + margin:
        return True
    else:
        return False




class Simulation:
    """
    Class holding all the functions and parameters for a single simulation instance. All quantities must be in physical SI units.
    Only the visuals instance deals with pixel dimensions.
    """

    def __init__(self, bt):
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        self.score = 0  # Initialisation of fitness score for this particular simulation
        self.t = 0  # Initialisation of time [s]

        # Lists holding simulated entities
        self.entities = []
        self.trees = []
        self.fruits = []
        self.drones = []

        self.n0_drones = N_DRONES
        self.n_rows = random.choice([3, 4, 5])

        self.bt = bt  
        
        self.load_environment()

        if VISUALISE:
            self.visuals = Visuals(WIDTH, HEIGHT)

    def load_environment(self):
        """
        loads simulated environment by placing trees, beetles and drones.
        :return: None
        """


        # Initial random placement of trees on map
        for i in range(self.n_rows):
            for j in range(int(round(N_TREES_PER_ROW * noise(NOISE), 0))):
                placing = True
                while placing:
                    y = HEIGHT / (self.n_rows + 1) * (i+1)
                    x = random.uniform(LAUNCHPAD_FRAC*WIDTH + R_TREE_MAX, WIDTH - R_TREE_MAX)

                    newtree = Entity('Tree ' + str(j + i * N_TREES_PER_ROW), 'tree', x, y, np.random.normal(R_TREE_AVG, R_TREE_STD))
                    if not any([check_collision(newtree, entity, -0.3) for entity in self.entities]):
                        self.entities.append(newtree)
                        self.trees.append(newtree)
                        placing = False


        # # Initial placement of fruit
        for tree in self.trees:
            if random.uniform(0, 1) < FRUIT_PROB:
                angle = random.uniform(-np.pi, np.pi)
                x = tree.x + tree.r_col * np.cos(angle)
                y = tree.y + tree.r_col * np.sin(angle)
                z = random.uniform(FRUIT_MIN_HEIGHT, TREE_HEIGHT)
                
                newfruit = Fruit('Fruit of ' + tree.name, x, y, z)
                
                if not any([check_collision(newfruit, othertree) for othertree in self.trees if not othertree.name == tree.name]):
                    self.fruits.append(newfruit)

       

        # Initial random placement of drones on launchpad (fraction of total map)
        for k in range(int(round(N_DRONES * noise(NOISE), 0))):
            placing = True
            while placing:
                x = (np.random.random() * WIDTH * LAUNCHPAD_FRAC * noise(NOISE))
                y = random.uniform(k * HEIGHT / N_DRONES, (k+1) * HEIGHT / N_DRONES)

                newdrone = Drone('Drone ' + str(k), 'drone', x, y, self.bt)
                if not any([check_collision(newdrone, entity) for entity in self.entities]):
                    self.entities.append(newdrone)
                    self.drones.append(newdrone)
                    # print(newdrone.name, 'placed!')
                    placing = False
            self.n0_drones = len(self.drones)

    def evaluate(self):
        """
        called when simulation is over to evaluate the fitness of a solution using three transfer functions: one for each criterion.
        :return: (list): 1. score for this particular simulation, lies in interval [0, 1], 2. fraction of killed drones,
                         3. fraction of killed beetles, 4. fraction of passed time.
        """
        scores = []
        for fruit in self.fruits:
            scores.append(1 - 3 *np.mean([item**2 / self.t**2 for item in fruit.record]))
        
        
        return np.mean(scores)

    def run(self):
        """
        loads environment, starts simulation loop and finally calls evaluation function.
        :return: (float) score for this particular simulation, lies in interval [0, 1].
        """
        running = True
        while running:
            # Print time and seed every 10s
            if int(round(self.t, 0)) % 10 == 0 and abs(int(round(self.t, 0)) - self.t) < 0.001:
                print('Seed:', SEED, 'Time:', round(self.t, 0), 's')
            
            for fruit in self.fruits:
                fruit.advance()

            # Drone simulation
            for drone in self.drones:
                for otherdrone in self.drones:
                    if check_collision(drone, otherdrone):
                        if drone in self.entities:
                            self.entities.remove(drone)
                        if drone in self.drones:
                            self.drones.remove(drone)  
                        self.entities.remove(otherdrone)
                        self.drones.remove(otherdrone)


                for tree in self.trees:
                    if check_collision(drone, tree) and drone.z <= TREE_HEIGHT:
                        self.drones.remove(drone)
                        self.entities.remove(drone)


                

                # # Maintain list of visible entities
                # for entity in self.entities:
                #     if drone.sees(entity) and entity not in drone.visible_entities:
                #         drone.visible_entities.append(entity)
                #     if not drone.sees(entity) and entity in drone.visible_entities:
                #         drone.visible_entities.remove(entity)

                drone.codrones = [otherdrone for otherdrone in self.drones if not otherdrone == drone]
                drone.advance()                    

                if not 0 < drone.x < WIDTH or not 0 < drone.y < HEIGHT:
                    self.entities.remove(drone)
                    self.drones.remove(drone)

                # for entity in drone.visible_entities:
                #     if entity not in self.entities:
                #         drone.visible_entities.remove(entity)
            
            print()
            # Update screen if requested
            if VISUALISE:
                self.visuals.update(self.trees, self.fruits, self.drones, self.t)
            
        

            # Add time step
            self.t += DT

            # End conditions: 80% of drones dead, all beetles dead or time up.
            if not self.drones or self.t >= T_MAX:
                running = False
                self.score = self.evaluate()
                print(self.score)

        return self.score
