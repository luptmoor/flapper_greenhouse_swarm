import numpy as np
from Entity import Entity
from Drone import Drone
from Fruit import Fruit
from Visuals import Visuals
from settings import *
import random
import torch
import pygame
import FlapperModel
from numba import jit, njit
import torch.nn as nn
import torch.nn.functional as F


class SwarmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 32)  # First hidden layer
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer
        self.fc3 = nn.Linear(16, 4)   # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))   # sigmoid to keep outputs bounded in (0, 1)

        # Map to correct ranges
        min_tensor = torch.tensor([-V_BACKWARD_MAX, -V_DOWN_MAX, -YAWRATE_MAX, -1])
        max_tensor = torch.tensor([V_FORWARD_MAX, V_UP_MAX, YAWRATE_MAX, 1])

        x = min_tensor + (max_tensor - min_tensor) * x
        x = x.detach().numpy()

        return x[0], x[1], x[2], x[3]  # vx, vz, r, msg


swarm_net = SwarmNet();

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


drone_dtype = np.dtype([
    ('active', 'b'),
    ('x', 'f8'),
    ('y', 'f8'),
    ('z', 'f8'),
    ('heading', 'f8'),
    ('vx', 'f8'),
    ('vz', 'f8'),
    ('r', 'f8'),
    ('mem', 'f8'),
    ('msg', 'f8'),
    ('swarm_matrix', 'f8', (N_DRONES, 4)),
])

drones = np.zeros(N_DRONES, dtype=drone_dtype)

def drone_sees(drone, entity):
    
    if entity is None:
        return False
    
    dx = drone['x'] - entity.x
    dy = drone['y'] - entity.y
    dz = drone['z'] - entity.z
    dh = np.sqrt(dx ** 2 + dy ** 2)  # Pythagoras

    elevation = np.atan(dz / dh)
    bearing = np.atan2(dy, dx) + np.pi          
    if bearing > np.pi: bearing -= 2*np.pi
    if bearing <-np.pi: bearing += 2*np.pi
    azimuth =  bearing - drone['heading']

    if dh <= R_TREE_AVG and np.abs(elevation) <= CAMERA_VFOV and np.abs(azimuth) <= CAMERA_HFOV:
        return True
    else:
        return False

@njit
def update_swarm_matrices():
    for i in range(N_DRONES):
        cpsi = np.cos(drones[i]['heading'])
        spsi = np.sin(drones[i]['heading'])
        R = np.array([
            [cpsi, -spsi, 0.0],
            [spsi,  cpsi, 0.0],
            [0.0,    0.0, 1.0]
        ])

        for j in range(N_DRONES):
            if i != j and drones[j]['active']:
                relpos = np.array([
                    drones[i]['x'] - drones[j]['x'],
                    drones[i]['y'] - drones[j]['y'],
                    drones[i]['z'] - drones[j]['z']
                ]) @ R
                drones[i]['swarm_matrix'][j] = [*relpos, drones[j]['msg']]
            else:
                drones[i]['swarm_matrix'][j] = [0.0, 0.0, 0.0, 0.0]


def drone_advance(drone):

    update_swarm_matrices();

    # if not (MANUAL and drone['entity']['name'] == 'Drone 0'):
    #     drone['message'] = msg
    # else:
    #     vx_cmd = vz_cmd = r_cmd = 0.0
    
    vx_cmd, vz_cmd, r_cmd, drone['msg'] = swarm_net.forward(torch.tensor(drone['swarm_matrix'], dtype=torch.float32).flatten())
    # , drone['mem']

    #print(drone['swarm_matrix'])

    model_state = FlapperModel.advance(vx_cmd, vz_cmd, r_cmd, DT)
    y = FlapperModel.to_output(model_state)

    drone['vx'] = y[0]
    drone['vz'] = y[1]
    drone['r'] = y[4]

    drone['heading'] += drone['r'] * DT
    drone['heading'] = (drone['heading'] + np.pi) % (2 * np.pi) - np.pi

    drone['x'] += drone['vx'] * DT * np.cos(drone['heading'])
    drone['y'] += drone['vx'] * DT * np.sin(drone['heading'])
    drone['z'] += drone['vz'] * DT

    drone['x'] = min(max(drone['x'], 0.01), WIDTH)
    drone['y'] = min(max(drone['y'], 0.01), HEIGHT)
    drone['z'] = min(max(drone['z'], 0.01), CEILING)



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
        for k in range(N_DRONES):
            x = (np.random.random() * WIDTH * LAUNCHPAD_FRAC)
            y = random.uniform(k * HEIGHT / N_DRONES, (k+1) * HEIGHT / N_DRONES)

            drones[k]['x'] = x;
            drones[k]['y'] = y;
            drones[k]['z'] = 0.01;
            drones[k]['heading'] = np.random.uniform(-np.pi, np.pi)
            drones[k]['active'] = True
        
        self.n0_drones = N_DRONES

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
        #dummy = input('Press enter to start')
        while running:
            # Print time and seed every 10s
            #if int(round(self.t, 0)) % 10 == 0 and abs(int(round(self.t, 0)) - self.t) < 0.001:
                #print('Seed:', SEED, 'Time:', round(self.t, 0), 's')
            
            for fruit in self.fruits:
                fruit.advance()

            # Drone simulation, TODO consider order of drones
            for i in range(N_DRONES):
                # for otherdrone in self.drones:
                #     if check_collision(drone, otherdrone):
                #         if drone in self.entities:
                #             self.entities.remove(drone)
                #         if drone in self.drones:
                #             self.drones.remove(drone)  
                #         self.entities.remove(otherdrone)
                #         self.drones.remove(otherdrone)


                for fruit in self.fruits:
                    if drone_sees(drones[i], fruit):
                        #print(f'{drone.name} sees {fruit.name}.')
                        fruit.reset_counter()
                        #drone.inspect(fruit)


                

                # # Maintain list of visible entities
                # for entity in self.entities:
                #     if drone.sees(entity) and entity not in drone.visible_entities:
                #         drone.visible_entities.append(entity)
                #     if not drone.sees(entity) and entity in drone.visible_entities:
                #         drone.visible_entities.remove(entity)

                # drone.codrones = [otherdrone for otherdrone in self.drones if not otherdrone == drone]

                # if drone.name == 'Drone 0' and MANUAL:
                #     for event in pygame.event.get():
                #         if event.type == pygame.KEYDOWN:
                #             if event.key == pygame.K_w:
                #                 print('fwd')
                #                 drone.vx = 0.2
                #             elif event.key == pygame.K_d:
                #                 drone.r = np.pi / 5
                #                 print('r')
                #             elif event.key == pygame.K_a:
                #                 drone.r = -np.pi / 5
                #                 print('l')
                #             elif event.key == pygame.K_SPACE:
                #                 drone.vz = 0.2
                #                 print('up')
                #             elif event.key == pygame.K_LSHIFT:
                #                 drone.vz = -0.2
                #                 print('dn')
                #         elif event.type == pygame.KEYUP:
                #             drone.vz = 0
                #             drone.vx = 0
                #             drone.r = 0

                drone_advance(drones[i])
            
                
                    
                # for entity in drone.visible_entities:
                #     if entity not in self.entities:
                #         drone.visible_entities.remove(entity)
            
            #print()
            # Update screen if requested
            if VISUALISE:
                self.visuals.update(self.trees, self.fruits, drones, self.t)
            
        

            # Add time step
            self.t += DT

            # End conditions: 80% of drones dead, all beetles dead or time up.
            if self.t >= T_MAX:
                running = False
                self.score = self.evaluate()
                print(self.score)

        return self.score