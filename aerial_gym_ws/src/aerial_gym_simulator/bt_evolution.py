
from PIL import Image
from aerial_gym.utils.logging import CustomLogger
logger = CustomLogger(__name__)
from behaviour_tree import BehaviourTree
from aerial_gym.utils.logging import CustomLogger
import torch
import numpy as np
import random
from settings import *


class BTEvolution:
    def __init__(self, simulator, population_size, n_generations, tmax, filepath):
        self.simulator = simulator
        self.simtime = tmax
        self.population_size = population_size
        self.n_generations = n_generations

        self.current_gen = self.population_size * [None]  # list of BTs
        self.current_fitnesses = self.population_size * [0]
        self.generation_counter = 0

        self.filepath = filepath

        # Initialise population
        # for i in range(self.population_size):
        #     self.current_gen[i] = BehaviourTree(random=True)

        # print(self.current_gen)

    def evaluate(self, x):
        fitness = x**2
        return fitness

    def save2csv(self):
        return 0

    def mutate(self):
        return 0

    def crossover(self):
        return 0
    
    def next_gen(self):
        # Sort based on fitness

        # random.shuffle(self.current_gen)  # Shuffle to ensure randomness
        # tournaments = [list(sublist) for sublist in np.array_split(self.current_gen, N_TOURNAMENTS)]

        # next_generation = [phenotype for _, phenotype in sorted(zip(self.current_fitnesses, self.current_gen), reverse=True)]

        # next_generation[N_ELITE:] = 

        return 0
    


    def simulate_current_gen(self):
        
        # actions: reference positions in local coordinate system
        actions = torch.zeros((self.simulator.num_envs, 4)).to("cuda:0")

        # Start simulation
        self.simulator.reset()

        # Simulate for 10,000 steps = 100s physical time,  80s computational time
        for i in range(self.simtime):
            if i % 1000 == 0:


                # Write to blackboard


                #actions = bt.feed_forward(blackboard)


                logger.info(f"Step {i}, changing target setpoint.")
                self.simulator.reset()
            self.simulator.step(actions=actions)

        fitness = self.evaluate(3);
        return fitness;

    def simulate_bt(self, bt, tmax):
        
        # actions: reference positions in local coordinate system
        actions = torch.zeros((self.simulator.num_envs, 4)).to("cuda:0")

        # Start simulation
        self.simulator.reset()

        depth_frames = []

        # Simulate for 10,000 steps = 100s physical time,  80s computational time
        for i in range(tmax):
            if i % 100 == 0:
                print()
                print(f'Step {i//100}')
                # Write to blackboard
                blackboard = {
                    "elapsed_battery_time": 0,
                    "absolute_x": 0,
                    "absolute_y": 0,
                    "absolute_z": 0,
                    "heading": 0,
                    "visit_list": 0,
                    "tof": 0,
                    "peer_distances": [],
                    "fruit_visible": False,
                    'swarmneural': 0,
                    'memory': 0.2,
                    'tofinput': torch.randn(1, 64),
                    'swarminput': torch.randn(1, 20)
                }
                

                observations = self.simulator.get_obs()
                print(f"Observed crashes: {observations['crashes']}")
                # for j in range(self.simulator.num_envs):
                #     actions[j, :] = bt.feed_forward(blackboard)
                
                #dof_pos = 20 / 57.3 * torch.ones((self.simulator.num_envs, 4)).to("cuda:0")
                # angle = 40
                # dof_pos = torch.tensor([-angle / 57.3, angle / 57.3, -angle / 57.3, angle / 57.3]).repeat(self.simulator.num_envs, 1)
                # self.simulator.robot_manager.robot.set_dof_velocity_targets(dof_pos)

                logger.info(f"Step {i}, changing target setpoint.")
            self.simulator.step(actions=actions)
            self.simulator.render(render_components="sensors")

        
            try:
            # define a normal direction to take a dot product with
                one_vec = torch.zeros_like(self.simulator.global_tensor_dict["depth_range_pixels"])
                one_vec[..., 0] = 1.0
                one_vec[..., 1] = 1 / 2.0
                one_vec[..., 2] = 1 / 3.0
                one_vec[:] = one_vec / torch.norm(one_vec, dim=-1, keepdim=True)
                cosine_vec = torch.abs(
                    torch.sum(one_vec * self.simulator.global_tensor_dict["depth_range_pixels"], dim=-1)
                )
                # max_dr = torch.max(cosine_vec)
                # min_dr = torch.min(cosine_vec)

                # print(torch.mean(cosine_vec), max_dr, min_dr)
                image1 = (255.0 * cosine_vec)[0, 0].cpu().numpy().astype(np.uint8)
            except Exception as e:
                raise e
            
            depth_image1 = Image.fromarray(image1)
            depth_frames.append(depth_image1)


        depth_frames[0].save(
            f"depth_frames_{i}.gif",
            save_all=True,
            append_images=depth_frames[1:],
            duration=100,
            loop=0,
        )




        fitness = self.evaluate(3);
        return fitness;