
from aerial_gym.utils.logging import CustomLogger
logger = CustomLogger(__name__)
from behaviour_tree import BehaviourTree
from aerial_gym.utils.logging import CustomLogger
import torch


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

        # Simulate for 10,000 steps = 100s physical time,  80s computational time
        for i in range(tmax):
            if i % 400 == 0:


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
                }
                
                for j in range(self.simulator.num_envs):
                    actions[j, :] = bt.feed_forward(blackboard)


                logger.info(f"Step {i}, changing target setpoint.")
            self.simulator.step(actions=actions)

        fitness = self.evaluate(3);
        return fitness;