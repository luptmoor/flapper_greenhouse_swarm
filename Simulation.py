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
from FlapperModel import advance_dynamics
from numba import jit, njit
import torch.nn as nn
import torch.nn.functional as F


class SwarmNet(nn.Module):
    def __init__(self):
        super().__init__()
        # In N x (N*4) = N x 20
        self.fc1 = nn.Linear(N_DRONES * 4, 32)  # First hidden layer
        # N x 16
        self.fc2 = nn.Linear(32, 16)  # Second hidden layer
        # N x 16
        self.fc3 = nn.Linear(16, 5)   # Output layer
        # N x 5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))   # sigmoid to keep outputs bounded in (0, 1)

        # Map to correct ranges
        min_tensor = torch.tensor([-V_BACKWARD_MAX, -V_DOWN_MAX, -YAWRATE_MAX, -1.0, -1.0])
        max_tensor = torch.tensor([V_FORWARD_MAX, V_UP_MAX, YAWRATE_MAX, 1.0, 1.0])

        x = min_tensor + (max_tensor - min_tensor) * x
        x = x.detach().numpy()

        return x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]  # vx, vz, r, msg, mem


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


active_array = np.ones(N_DRONES, dtype=np.bool)
x_array = np.random.uniform(0.1, WIDTH * LAUNCHPAD_FRAC, N_DRONES).astype(np.float32)
y_array = np.random.uniform(0.1, HEIGHT / N_DRONES, N_DRONES).astype(np.float32) + np.arange(N_DRONES) * HEIGHT / N_DRONES
z_array = 0.01 * np.ones(N_DRONES, dtype=np.float32)
heading_array = np.random.uniform(-np.pi, np.pi, N_DRONES).astype(np.float32)
vx_array = np.zeros(N_DRONES, dtype=np.float32)
vz_array = np.zeros(N_DRONES, dtype=np.float32)
r_array = np.zeros(N_DRONES, dtype=np.float32)
mem_array = np.zeros(N_DRONES, dtype=np.float32)
msg_array = np.zeros(N_DRONES, dtype=np.float32)
vxcmd_array = np.zeros(N_DRONES, dtype=np.float32)
vzcmd_array = np.zeros(N_DRONES, dtype=np.float32)
rcmd_array = np.zeros(N_DRONES, dtype=np.float32)
swarm_array = np.zeros((N_DRONES, N_DRONES*4), dtype=np.float32)


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
def update_swarm_matrices(x_array, y_array, z_array, heading_array, msg_array, swarm_array):
    cpsi = np.cos(heading_array)
    spsi = np.sin(heading_array)

    dx = x_array[:, None] - x_array[None, :]
    dy = y_array[:, None] - y_array[None, :]
    dz = z_array[:, None] - z_array[None, :]

    dx_rot = dx * cpsi + dy * spsi
    dy_rot = dy * -spsi + dy * cpsi

    msg_mat = np.empty((N_DRONES, N_DRONES), dtype=np.float32)
    for i in range(N_DRONES):
        msg_mat[i, :] = msg_array
        msg_mat[i, i] = 0.0

    for j in range(N_DRONES):
        swarm_array[:, 4*j + 0] = dx_rot[:, j]
        swarm_array[:, 4*j + 1] = dy_rot[:, j]
        swarm_array[:, 4*j + 2] = dz[:, j]
        swarm_array[:, 4*j + 3] = msg_mat[:, j]


@njit
def advance_dynamics(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, vxcmd_array, vzcmd_array, rcmd_array):
    # if not (MANUAL and drone['entity']['name'] == 'Drone 0'):
    #     drone['message'] = msg
    # else:
    #     vx_cmd = vz_cmd = r_cmd = 0.0
    
    # , drone['mem']

    #print(drone['swarm_matrix'])

    # First order lag
    vx_array[:] = (1 - DT/TAU_VX) * vx_array + DT/TAU_VX * vxcmd_array;
    vz_array[:] = (1 - DT/TAU_VZ) * vz_array + DT/TAU_VZ * vzcmd_array;
    r_array[:] = (1 - DT/TAU_R) * r_array + DT/TAU_R * rcmd_array;

    heading_array[:] += r_array * DT
    heading_array[:] = (heading_array + np.pi) % (2 * np.pi) - np.pi

    x_array[:] = np.clip(x_array + vx_array * DT * np.cos(heading_array), 0.01, WIDTH)
    y_array[:] = np.clip(y_array + vx_array * DT * np.sin(heading_array), 0.01, HEIGHT)
    z_array[:] = np.clip(z_array + vz_array * DT, 0.01, CEILING)


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
        global x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, mem_array, msg_array, vxcmd_array, vzcmd_array, rcmd_array, swarm_array
        running = True
        #dummy = input('Press enter to start')
        while running:
            # Print time and seed every 10s
            #if int(round(self.t, 0)) % 10 == 0 and abs(int(round(self.t, 0)) - self.t) < 0.001:
                #print('Seed:', SEED, 'Time:', round(self.t, 0), 's')
            
            for fruit in self.fruits:
                fruit.advance()

            # Drone simulation, TODO consider order of drones
            update_swarm_matrices(x_array, y_array, z_array, heading_array, msg_array, swarm_array);
            #print(drones[i]['swarm_matrix'])
            vxcmd_array, vzcmd_array, rcmd_array, msg_array, mem_array = swarm_net.forward(torch.Tensor(swarm_array))
            #drones[i]['vx'], drones[i]['vz'], drones[i]['r'] = advance_dynamics(drones[i]['vx_cmd'], drones[i]['vz_cmd'], drones[i]['r_cmd'], DT)
            advance_dynamics(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, vxcmd_array, vzcmd_array, rcmd_array)
            #print(drones[i]['vx_cmd'], drones[i]['vz_cmd'], drones[i]['r_cmd'])


            # for i in range(N_DRONES):
            #     for fruit in self.fruits:
            #         if drone_sees(drones[i], fruit):
            #             #print(f'{drone.name} sees {fruit.name}.')
            #             fruit.reset_counter()
            #             #drone.inspect(fruit)
                
                

                

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
            
                
                    
                # for entity in drone.visible_entities:
                #     if entity not in self.entities:
                #         drone.visible_entities.remove(entity)
            
            #print()
            # Update screen if requested
            if VISUALISE:
                self.visuals.update(self.trees, self.fruits, x_array, y_array, z_array, heading_array, active_array, self.t)
            
        

            # Add time step
            self.t += DT

            # End conditions: 80% of drones dead, all beetles dead or time up.
            if self.t >= T_MAX:
                running = False
                self.score = self.evaluate()
                print(self.score)

        return self.score