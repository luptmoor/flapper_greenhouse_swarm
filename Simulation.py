import numpy as np
from Entity import Entity
from Fruit import Fruit
from Visuals import Visuals
from settings import *
import random
import torch
import pygame
from numba import jit, njit
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os


class WeightedDeepSet(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=16, out_dim=5):
        super().__init__()

        # Preparation
        self.reshape = nn.Sequential(
            nn.Unflatten(1, (N_DRONES-1, 4))
        )
        
        # Embedding
        self.q = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # to output
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, tensor):
        # positions: [n_points, 3]
        # weights: [n_points, 1]
        
        # in: N x 4(N-1) = N x 16
        x = self.reshape(tensor) # N x 4 x 4
        coords  = x[:, :, :3] # N x 4 x 3
        weights = x[:, :, -1].unsqueeze(-1) # N x 4 x 1

        embedded = self.q(coords)       # N x 4 x 16
        weighted = embedded * weights      # N x 4 x 16
        pooled = weighted.sum(dim=1)      # N x 16
        
        raw_output = self.rho(pooled)     # N x 5
        x = torch.sigmoid(raw_output)     # N x 5
        

        # Map to correct ranges
        min_tensor = torch.tensor([-V_BACKWARD_MAX, -V_DOWN_MAX, -YAWRATE_MAX, 1.0, -1.0])
        max_tensor = torch.tensor([V_FORWARD_MAX, V_UP_MAX, YAWRATE_MAX, 2.0, 1.0])

        x = min_tensor + (max_tensor - min_tensor) * x
        x = x.detach().numpy()

        return x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]  # vx, vz, r, msg, mem







class SwarmNet(nn.Module):
    def __init__(self):
        super().__init__()
        # In N x (N*4) = N x 20
        self.fc1 = nn.Linear((N_DRONES-1) * 4, 32)  # First hidden layer
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


# def check_fruit_discoveries(x_array, y_array, z_array, heading_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array):
#     dx = x_array[:, np.newaxis] - fruit_x_array[np.newaxis, :]
#     dy = y_array[:, np.newaxis] - fruit_y_array[np.newaxis, :]
#     dz = z_array[:, np.newaxis] - fruit_z_array[np.newaxis, :]
#     dh = np.sqrt(dx ** 2 + dy ** 2)  # Pythagoras

#     elevation = np.atan(dz / dh)
#     bearing = np.atan2(dy, dx) + np.pi

#     bearing[bearing > np.pi] -= 2*np.pi
#     bearing[bearing <-np.pi] += 2*np.pi

#     azimuth = bearing - heading_array[:, np.newaxis]

#     drone_side_array = heading_array >= 0

#     discovery_array = (
#         (dh <= R_DISCOVERY) &
#         (np.abs(elevation) <= CAMERA_VFOV) &
#         (np.abs(azimuth) <= CAMERA_HFOV) &
#         (drone_side_array[:, np.newaxis] == fruit_side_array[np.newaxis, :])
#     ) # 5 x 30

#     fruit_disc_array[:] = (np.sum(discovery_array, axis=0) >= 1) | fruit_disc_array

@njit
def check_fruit_discoveries(
    x_array, y_array, z_array, heading_array,
    fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array,
    fruit_disc_array
):
    num_drones = x_array.shape[0]
    num_fruits = fruit_x_array.shape[0]

    for i in range(num_drones):
        for j in range(num_fruits):
            dx = x_array[i] - fruit_x_array[j]
            dy = y_array[i] - fruit_y_array[j]
            dz = z_array[i] - fruit_z_array[j]

            dh = np.sqrt(dx ** 2 + dy ** 2)
            if dh == 0.0:
                continue  # avoid division by zero

            elevation = np.arctan(dz / dh)
            bearing = np.arctan2(dy, dx) + np.pi

            if bearing > np.pi:
                bearing -= 2 * np.pi
            if bearing < -np.pi:
                bearing += 2 * np.pi

            azimuth = bearing - heading_array[i]
            if azimuth > np.pi:
                azimuth -= 2 * np.pi
            if azimuth < -np.pi:
                azimuth += 2 * np.pi

            drone_side = heading_array[i] >= 0.0

            if (dh <= R_DISCOVERY and
                abs(elevation) <= CAMERA_VFOV and
                abs(azimuth) <= CAMERA_HFOV and
                drone_side == fruit_side_array[j]):
                fruit_disc_array[j] = True


@njit
def update_swarm_matrices(x_array, y_array, z_array, heading_array, msg_array, swarm_array, active_array):
    cpsi = np.cos(heading_array)
    spsi = np.sin(heading_array)

    # 3 N x N arrays of distances with zero diagonal
    dx = x_array[:, None] - x_array[None, :]
    dy = y_array[:, None] - y_array[None, :]
    dz = z_array[:, None] - z_array[None, :]

    # Rotate by heading
    dx_rot = dx *  cpsi + dy * spsi
    dy_rot = dx * -spsi + dy * cpsi

    # Create an N x 4(N-1) matrix
    for i in range(N_DRONES):
        for j in range(N_DRONES-1):
            idx = 4 * j  # Starting index for each drone's set of 4 columns

            if j >= i: j += 1
            j = j % N_DRONES
            
            # Update swarm array with rotated distances and messages
            swarm_array[i, idx + 0] = dx_rot[i, j] * active_array[j]  # dx
            swarm_array[i, idx + 1] = dy_rot[i, j] * active_array[j]  # dy
            swarm_array[i, idx + 2] = dz[i, j] * active_array[j]      # dz
            swarm_array[i, idx + 3] = msg_array[j] * active_array[j]  # message


@njit
def advance_dynamics(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, vxcmd_array, vzcmd_array, rcmd_array, active_array):

    # First order lag
    vx_array[:] = (1 - DT/TAU_VX) * vx_array + DT/TAU_VX * vxcmd_array;
    vz_array[:] = (1 - DT/TAU_VZ) * vz_array + DT/TAU_VZ * vzcmd_array;
    r_array[:] = (1 - DT/TAU_R) * r_array + DT/TAU_R * rcmd_array;

    heading_array[:] += r_array * DT
    heading_array[:] = (heading_array + np.pi) % (2 * np.pi) - np.pi

    x_array[:] = np.clip((x_array + vx_array * DT * np.cos(heading_array)) * active_array, 0.2, WIDTH)
    y_array[:] = np.clip((y_array + vx_array * DT * np.sin(heading_array)) * active_array, 0.2, HEIGHT)
    z_array[:] = np.clip((z_array + vz_array * DT)                         * active_array, 0.2, CEILING)


@njit
def check_drone_collisions(x_array, y_array, z_array, active_array):
    dx = x_array[np.newaxis, :] - x_array[:, np.newaxis]
    dy = y_array[np.newaxis, :] - y_array[:, np.newaxis]
    dz = z_array[np.newaxis, :] - z_array[:, np.newaxis]

    distance = np.sqrt(np.pow(dx, 2) + np.pow(dy, 2) + np.pow(dz, 2))
    collision_free = distance > 2.2*R_DRONE
 
    active_array[:] = (np.sum(collision_free, axis=1) >= (N_DRONES-1)) * active_array

@njit
def envelope_ellipse(z):
    return V_UP_MAX / SQUEEZE_RANGE * np.sqrt(SQUEEZE_RANGE**2 - (z - SQUEEZE_RANGE)**2)

@njit
def squeeze_vertical_speed(z_array, vzcmd_array):
    for i in range(N_DRONES):
        if z_array[i] + SQUEEZE_RANGE >= CEILING:
            vzcmd_array[i] = min(vzcmd_array[i], envelope_ellipse(CEILING - z_array[i]))

        elif z_array[i] <= SQUEEZE_RANGE + 0.2:
            vzcmd_array[i] = max(vzcmd_array[i], -envelope_ellipse(z_array[i] - 0.2))
    
    return vzcmd_array


class Simulation:
    """
    Class holding all the functions and parameters for a single simulation instance. All quantities must be in physical SI units.
    Only the visuals instance deals with pixel dimensions.
    """

    def __init__(self, vis, seed=SEED):
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Lists holding simulated entities
        self.entities = []
        self.trees = []

        self.n_rows = random.choice([3, 4, 5])
        
        if vis:
            self.visuals = Visuals(WIDTH, HEIGHT)

    def load_environment(self, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array):
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


        # Initial placement of fruit
        f = 0;
        while f < N_FRUIT:
            for tree in self.trees:
                if random.uniform(0, 1) < FRUIT_PROB:
                    angle = random.uniform(-np.pi, np.pi)
                    x = tree.x + tree.r_col * np.cos(angle)
                    y = tree.y + tree.r_col * np.sin(angle)
                    z = random.uniform(FRUIT_MIN_HEIGHT, TREE_HEIGHT)
                    newfruit = Fruit(f'Fruit {f}', x, y, z)
                    
                    if not any([check_collision(newfruit, othertree) for othertree in self.trees if not othertree.name == tree.name]):
                        fruit_x_array[f] = x;
                        fruit_y_array[f] = y;
                        fruit_z_array[f] = z;
                        fruit_side_array[f] = (angle >= 0);
                        f += 1;
                        if f >= N_FRUIT: break

        return fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array

                
        
def run(sim, swarm_net, vis, gen):
    """
    loads environment, starts simulation loop and finally calls evaluation function.
    :return: (float) score for this particular simulation, lies in interval [0, 1].
    """
    t = 0.0;
    score = 0.0;

    active_array = np.ones(N_DRONES, dtype=np.bool)
    x_array = np.random.uniform(0.1, WIDTH * LAUNCHPAD_FRAC, N_DRONES).astype(np.float32)
    y_array = np.random.uniform(0.1, HEIGHT / N_DRONES, N_DRONES).astype(np.float32) + np.arange(N_DRONES) * HEIGHT / N_DRONES
    z_array = 0.01 * np.ones(N_DRONES, dtype=np.float32)
    heading_array = np.random.uniform(-np.pi, np.pi, N_DRONES).astype(np.float32)
    vx_array = np.zeros(N_DRONES, dtype=np.float32)
    vz_array = np.zeros(N_DRONES, dtype=np.float32)
    r_array = np.zeros(N_DRONES, dtype=np.float32)
    mem_array = np.zeros(N_DRONES, dtype=np.float32)
    msg_array = np.random.uniform(1.0, 2.0, (N_DRONES, MAX_TICKS)).astype(np.float32)
    vxcmd_array = np.zeros(N_DRONES, dtype=np.float32)
    vzcmd_array = np.zeros(N_DRONES, dtype=np.float32)
    rcmd_array = np.zeros(N_DRONES, dtype=np.float32)
    swarm_array = np.zeros((N_DRONES, (N_DRONES-1)*4), dtype=np.float32)

    fruit_x_array = np.zeros(N_FRUIT, dtype=np.float32)
    fruit_y_array = np.zeros(N_FRUIT, dtype=np.float32)
    fruit_z_array = np.zeros(N_FRUIT, dtype=np.float32)
    fruit_t_array = np.zeros((N_FRUIT, MAX_TICKS+1), dtype=np.float32)
    fruit_side_array = np.zeros(N_FRUIT, dtype=np.bool_)
    fruit_disc_array = np.zeros(N_FRUIT, dtype=np.bool_)

    fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array = sim.load_environment(fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array)



    #fig, axes, lines = live_plot_init(N_DRONES)


    for i in range(MAX_TICKS):
        fruit_t_array[:, i+1] = fruit_t_array[:, i] + DT

        # Drone simulation, TODO consider order of drones
        
        update_swarm_matrices(x_array, y_array, z_array, heading_array, msg_array[:, i], swarm_array, active_array);
        vxcmd_array, vzcmd_array, rcmd_array, msg_array[:, i], mem_array = swarm_net.forward(torch.Tensor(swarm_array))
        vzcmd_array = squeeze_vertical_speed(z_array, vzcmd_array)

        msg_array[:, i] = msg_array[:, i] * active_array
        mem_array[:] = mem_array * active_array

        advance_dynamics(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, vxcmd_array, vzcmd_array, rcmd_array, active_array)
     
        check_drone_collisions(x_array, y_array, z_array, active_array)
        check_fruit_discoveries(x_array, y_array, z_array, heading_array, fruit_x_array, fruit_y_array, fruit_z_array, fruit_side_array, fruit_disc_array)
        

        #live_plot_update(lines, msg_array[:, :i+1])


        # Update screen if requested
        if vis:
            sim.visuals.update(sim.trees, fruit_x_array, fruit_y_array, fruit_z_array, fruit_t_array, x_array, y_array, z_array, heading_array, active_array, fruit_disc_array, t, i)
        
        # Add time step
        t += DT

        if np.sum(active_array) < 2:
            break

        
    score = np.sum(active_array) / N_DRONES * np.sum(fruit_disc_array) / N_FRUIT
    print(score)

    if vis:
        folder = f"gen_{gen}"
        os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist
        filename = os.path.join(folder, f"{score}_d_{np.sum(active_array)}_f_{np.sum(fruit_disc_array)}_{np.random.uniform(0, 1):.2f}.png")
        plot_message_array(msg_array, filename)

    return score



def plot_message_array(msg_array, filename="message_array_plot.png"):
    N, T = msg_array.shape

    fig, axes = plt.subplots(N, 1, figsize=(10, 2 * N), sharex=True)

    if N == 1:
        axes = [axes]  # Ensure axes is always iterable

    for i in range(N):
        axes[i].plot(msg_array[i])
        axes[i].grid(True)
        axes[i].set_ylabel(f"Drone {i}")

    axes[-1].set_xlabel("Timestep")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    #plt.show()
    plt.close()
    print(f"[✔] Saved plot to {filename}")





def live_plot_init(N):
    plt.ion()
    fig, axes = plt.subplots(N, 1, figsize=(10, 2 * N), sharex=True)
    if N == 1:
        axes = [axes]

    lines = []
    for ax in axes:
        line, = ax.plot([], [])
        ax.grid(True)
        lines.append(line)

    plt.tight_layout()
    return fig, axes, lines

def live_plot_update(lines, data_array):
    for i, line in enumerate(lines):
        line.set_xdata(np.arange(data_array.shape[1]))
        line.set_ydata(data_array[i])
        line.axes.relim()
        line.axes.autoscale_view()

    plt.pause(0.01)  # allow GUI event loop to run