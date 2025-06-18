import copy
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

from io import BytesIO
from PIL import Image

import os


class WeightedDeepSet(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=16, out_dim=4):
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
        # coords  = x[:, :, :3] # N x 4 x 3
        # weights = x[:, :, -1].unsqueeze(-1) # N x 4 x 1

        embedded = self.q(x)       # N x 4 x 16
        #weighted = embedded * weights      # N x 4 x 16
        pooled = embedded.sum(dim=1)      # N x 16
        
        raw_output = self.rho(pooled)     # N x 5
        x = torch.sigmoid(raw_output)     # N x 5
        

        # Map to correct ranges
        min_tensor = torch.tensor([-V_BACKWARD_MAX, -V_DOWN_MAX, -YAWRATE_MAX, 1.0])
        max_tensor = torch.tensor([V_FORWARD_MAX, V_UP_MAX, YAWRATE_MAX, 2.0])

        x = min_tensor + (max_tensor - min_tensor) * x
        x = x.detach().numpy()

        return x[:, 0], x[:, 1], x[:, 2]**3 / YAWRATE_MAX**2, x[:, 3]  # vx, vz, r, msg






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



def update_swarm_matrices(x_array, y_array, z_array, swarm_array, active_array):
    # 3 N x N arrays of distances with zero diagonal
    dx = x_array[:, np.newaxis] - x_array[np.newaxis, :]
    dy = y_array[:, np.newaxis] - y_array[np.newaxis, :]
    dz = z_array[:, np.newaxis] - z_array[np.newaxis, :]

    # Create an N x 3(N-1) matrix
    for i in range(N_DRONES):
        for j in range(N_DRONES-1):
            idx = 3 * j  # Starting index for each drone's set of 4 columns

            if j >= i: j += 1
            j = j % N_DRONES
           
            # Update swarm array with rotated distances and messages
            swarm_array[i, idx + 0] = dx[i, j] * active_array[j]  # dx
            swarm_array[i, idx + 1] = dy[i, j] * active_array[j]  # dy
            swarm_array[i, idx + 2] = dz[i, j] * active_array[j]  # dz


@njit
def advance_dynamics(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, vxcmd_array, vzcmd_array, rcmd_array, active_array, obstacle_array, tick):

    # First order lag
    vx_array[:] = (1 - DT/TAU_VX) * vx_array + DT/TAU_VX * vxcmd_array
    vz_array[:] = (1 - DT/TAU_VZ) * vz_array + DT/TAU_VZ * vzcmd_array
    r_array[:] = (1 - DT/TAU_R) * r_array + DT/TAU_R * rcmd_array

    heading_array[:] += r_array * DT
    heading_array[:] = (heading_array + np.pi) % (2 * np.pi) - np.pi

    for i in range(N_DRONES):
        if active_array[i] == 0:
            continue

        if is_inside_obstacle(x_array[i, tick], y_array[i, tick], z_array[i, tick], obstacle_array):
            active_array[i] = 0
            #print(f'collision with obstacle for drone {i}')
        
        if (not 0.0 < x_array[i, tick] < WIDTH) or (not 0.0 < y_array[i, tick] < HEIGHT):
            active_array[i] = 0
            #print(f'out of bounds for drone {i}')

        if active_array[i] == 0:
            continue

        
        if tick < MAX_TICKS - 1:
            x_array[i, tick+1] = x_array[i, tick] + vx_array[i] * DT * np.cos(heading_array[i])
            y_array[i, tick+1] = y_array[i, tick] + vx_array[i] * DT * np.sin(heading_array[i])
            z_array[i, tick+1] = min(max(z_array[i, tick] + vz_array[i] * DT, 0.1), CEILING)


@njit
def is_inside_obstacle(x, y, z, obstacles):
    for i in range(obstacles.shape[0]):
        if (obstacles[i, 0] <= x <= obstacles[i, 0] + obstacles[i, 3] and
            obstacles[i, 1] <= y <= obstacles[i, 1] + obstacles[i, 4] and
            obstacles[i, 2] <= z <= obstacles[i, 2] + obstacles[i, 5]):
            return True
    return False


@njit
def check_drone_collisions(x_array, y_array, z_array, active_array):
    dx = x_array[np.newaxis, :] - x_array[:, np.newaxis]
    dy = y_array[np.newaxis, :] - y_array[:, np.newaxis]
    dz = z_array[np.newaxis, :] - z_array[:, np.newaxis]

    distance = np.sqrt(np.pow(dx, 2) + np.pow(dy, 2) + np.pow(dz, 2))
    collision_free = distance > 2.2*R_DRONE
 
    active_array[:] = (np.sum(collision_free, axis=1) >= (N_DRONES-1)) * active_array


@njit
def _envelope_ellipse(z):
    return V_UP_MAX / SQUEEZE_RANGE * np.sqrt(SQUEEZE_RANGE**2 - (z - SQUEEZE_RANGE)**2)

@njit
def squeeze_vertical_speed(z_array, vzcmd_array):
    for i in range(N_DRONES):
        if z_array[i] + SQUEEZE_RANGE >= CEILING:
            vzcmd_array[i] = min(vzcmd_array[i], _envelope_ellipse(CEILING - z_array[i]))

        elif z_array[i] <= SQUEEZE_RANGE + 0.3:
            vzcmd_array[i] = max(vzcmd_array[i], -_envelope_ellipse(z_array[i] - 0.3))
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
            self.visuals = Visuals(WIDTH, HEIGHT, self.n_rows)


    def load_environment(self, obstacle_array):
        """
        loads simulated environment by placing trees, beetles and drones.
        :return: None
        """

        # Initial random placement of trees on map
        for i in range(self.n_rows):
            # xA, yA, zA, width, height, depth
            obstacle_array[i, :] = np.array([LAUNCHPAD_FRAC*WIDTH + R_TREE_AVG, HEIGHT / (self.n_rows + 1) * (i+1) - 0.5*R_TREE_AVG, 0.0, (1-LAUNCHPAD_FRAC)*WIDTH - 2*R_TREE_AVG, R_TREE_AVG, TREE_HEIGHT])
            
        #     for j in range(int(round(N_TREES_PER_ROW * noise(NOISE), 0))):
        #         placing = True
        #         while placing:
        #             y = HEIGHT / (self.n_rows + 1) * (i+1)
        #             x = random.uniform(LAUNCHPAD_FRAC*WIDTH + R_TREE_MAX, WIDTH - R_TREE_MAX)

        #             newtree = Entity('Tree ' + str(j + i * N_TREES_PER_ROW), 'tree', x, y, np.random.normal(R_TREE_AVG, R_TREE_STD))
        #             if not any([check_collision(newtree, entity, -0.3) for entity in self.entities]):
        #                 self.entities.append(newtree)
        #                 self.trees.append(newtree)
        #                 placing = False
        

        # # Initial placement of fruit
        # f = 0;
        # while f < N_FRUIT:
        #     for tree in self.trees:
        #         if random.uniform(0, 1) < FRUIT_PROB:
        #             angle = random.uniform(-np.pi, np.pi)
        #             x = tree.x + tree.r_col * np.cos(angle)
        #             y = tree.y + tree.r_col * np.sin(angle)
        #             z = random.uniform(FRUIT_MIN_HEIGHT, TREE_HEIGHT)
        #             newfruit = Fruit(f'Fruit {f}', x, y, z)
                    
        #             if not any([check_collision(newfruit, othertree) for othertree in self.trees if not othertree.name == tree.name]):
        #                 fruit_x_array[f] = x;
        #                 fruit_y_array[f] = y;
        #                 fruit_z_array[f] = z;
        #                 fruit_side_array[f] = (angle >= 0);
        #                 f += 1;
        #                 if f >= N_FRUIT: break

        return obstacle_array

                
        
def run(sim, bt, vis=True, gen=1, phenotype=1):
    """
    loads environment, starts simulation loop and finally calls evaluation function.
    :return: (float) score for this particular simulation, lies in interval [0, 1].
    """
    t = 0.0;
    score = 0.0;

    active_array = np.ones(N_DRONES, dtype=np.bool)

    x_array = np.zeros((N_DRONES, MAX_TICKS), dtype=np.float32)
    y_array = np.zeros((N_DRONES, MAX_TICKS), dtype=np.float32)
    z_array = np.zeros((N_DRONES, MAX_TICKS), dtype=np.float32)
    x_array[:, 0] = np.random.uniform(0.1, WIDTH * LAUNCHPAD_FRAC, N_DRONES).astype(np.float32)
    y_array[:, 0] = np.random.uniform(0.1, HEIGHT / N_DRONES, N_DRONES).astype(np.float32) + np.arange(N_DRONES) * HEIGHT / N_DRONES
    z_array[:, 0] = 0.21 * np.ones(N_DRONES, dtype=np.float32)

    heading_array = np.random.uniform(-0.5, 0.5, N_DRONES).astype(np.float32)
    vx_array = np.zeros(N_DRONES, dtype=np.float32)
    vz_array = np.zeros(N_DRONES, dtype=np.float32)
    r_array = np.zeros(N_DRONES, dtype=np.float32)
    msg_array = np.zeros(N_DRONES, dtype=np.int8)
    vxcmd_array = np.zeros(N_DRONES, dtype=np.float32)
    vzcmd_array = np.zeros(N_DRONES, dtype=np.float32)
    rcmd_array = np.zeros(N_DRONES, dtype=np.float32)
    swarm_array = np.zeros((N_DRONES, (N_DRONES-1)*3), dtype=np.float32)
    string_array = np.zeros(N_DRONES, dtype=np.object_)

    skip = np.zeros(N_DRONES, dtype=np.bool_)

    # fruit_x_array = np.zeros(N_FRUIT, dtype=np.float32)
    # fruit_y_array = np.zeros(N_FRUIT, dtype=np.float32)
    # fruit_z_array = np.zeros(N_FRUIT, dtype=np.float32)
    # fruit_t_array = np.zeros((N_FRUIT, MAX_TICKS+1), dtype=np.float32)
    # fruit_side_array = np.zeros(N_FRUIT, dtype=np.bool_)
    # fruit_disc_array = np.zeros(N_FRUIT, dtype=np.bool_)

    obstacle_array = np.zeros((5, 6), dtype=np.float32)
    obstacle_array = sim.load_environment(obstacle_array)

    bt_list = [copy.deepcopy(bt) for _ in range(N_DRONES)]

    if SHOW_BT:
        bt_screen = init_bt_visualizer()
        update_bt_visualizer(bt_screen, bt_list[0])



    for i in range(MAX_TICKS):

        #update_swarm_matrices(x_array[:, i], y_array[:, i], z_array[:, i], swarm_array, active_array)

        if i % 10 == 0:
            for j in range(N_DRONES):
                vxcmd_array[j], vzcmd_array[j], rcmd_array[j], msg_array[j], string_array[j], skip[j] = bt_list[j].feed_forward(i, x_array[j, i], y_array[j, i], z_array[j, i], heading_array[j],
                                                                                        vx_array[j], vz_array[j], r_array[j],
                                                                                        swarm_array[j], 
                                                                                        obstacle_array, active_array, msg_array) 
            if REALTIME_BT: 
                update_bt_visualizer(bt_screen, bt_list[0])
                
            for j in range(N_DRONES):
                bt_list[j].root.reset()



        #vzcmd_array = squeeze_vertical_speed(z_array[:, i], vzcmd_array)
        advance_dynamics(x_array, y_array, z_array, heading_array, vx_array, vz_array, r_array, vxcmd_array, vzcmd_array, rcmd_array, active_array, obstacle_array, i)
        

        #check_drone_collisions(x_array[:, i], y_array[:, i], z_array[:, i], active_array)
        #check_fruit_discoveries(x_array[:, i], y_array[:, i], z_array[:, i], heading_array)
        
        t += DT


        if np.all(skip):
            break

        # Update screen if requested
        if vis:
            skip[:] = sim.visuals.update(sim.trees, x_array, y_array, z_array[:, i], heading_array, active_array, string_array, t, i)
            

        if np.sum(active_array) < 1:
            break

        

        
    score = calc_fitness(x_array, y_array, z_array)

    if vis:
        folder = f"gen_{gen}"
        os.makedirs(folder, exist_ok=True)  # create folder if it doesn't exist
        filename = os.path.join(folder, f"BT_{phenotype}_score_{score:.2}_traj.png")
        plot_3d_trajectory(x_array, y_array, z_array, filename=filename)


    if SHOW_BT:
        plt.ioff()
        plt.close(bt_screen[0])
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
    plt.show()
    plt.close()
    print(f"[✔] Saved plot to {filename}")




def plot_3d_trajectory(x_array, y_array, z_array, filename="3d_trajectory.png"):
    """
    Plot 3D trajectory of the drones
    :param x_array: Array of x coordinates with shape (N_DRONES, N_TICKS)
    :param y_array: Array of y coordinates with shape (N_DRONES, N_TICKS)
    :param z_array: Array of z coordinates with shape (N_DRONES, N_TICKS)"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N_DRONES):
        ax.plot(x_array[i], y_array[i], z_array[i], label=f'Drone {i}')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_zlim(0, CEILING)
    ax.legend()

    # indicate the voxel size (VOXEL_SIZE) with grid lines
    ax.set_xticks(np.arange(0, WIDTH, VOXEL_SIZE))
    ax.set_yticks(np.arange(0, HEIGHT, VOXEL_SIZE))
    ax.set_zticks(np.arange(0, CEILING, VOXEL_SIZE))
  
  
    plt.title("3D Trajectory of Drones")
    #plt.savefig(filename, dpi=150)
    #plt.show()

    plt.close()
    print(f"[✔] Saved 3D trajectory plot to {filename}")


def calc_fitness(x_array, y_array, z_array):
    """
    Discretize 3D space into voxels and count number of voxels visited by drones
    :param x_array: Array of x coordinates with shape (N_DRONES, N_TICKS)
    :param y_array: Array of y coordinates with shape (N_DRONES, N_TICKS)
    :param z_array: Array of z coordinates with shape (N_DRONES, N_TICKS)
    :return: fitness score
    """ 

    visited_voxels = set()
    for i in range(N_DRONES):
        for j in range(MAX_TICKS):
            voxel = (int(x_array[i, j] // VOXEL_SIZE), int(y_array[i, j] // VOXEL_SIZE), int(z_array[i, j] // VOXEL_SIZE))
            visited_voxels.add(voxel)

    fitness_score = len(visited_voxels) / (WIDTH * HEIGHT * CEILING / (VOXEL_SIZE ** 3))

    print(f"Visited voxels: {len(visited_voxels)} out of {int(WIDTH * HEIGHT * CEILING / (VOXEL_SIZE ** 3))} ({len(visited_voxels) / (WIDTH * HEIGHT * CEILING / (VOXEL_SIZE ** 3)):.2%})")
    return fitness_score








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




def init_bt_visualizer():
    """
    Initializes a live BT visualization window.
    Returns the figure and axis handles for future updates.
    """
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    return (fig, ax)


def update_bt_visualizer(window, bt):
    fig, ax = window

    # Render Graphviz image to in-memory buffer
    dot = bt.plot()
    png_data = dot.pipe(format='png')
    img = Image.open(BytesIO(png_data))

    # Display with Matplotlib
    ax.clear()
    ax.imshow(img)
    ax.axis('off')
    fig.canvas.draw()
    plt.pause(0.01)