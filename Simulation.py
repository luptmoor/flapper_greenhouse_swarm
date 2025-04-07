import numpy as np
from Entity import Entity
from Beetle import Beetle
from Drone import Drone
from Visuals import Visuals
from settings import *


def F_time(x):
    """
    Mathematical transfer function for time criterion.
    :param x: (float) function argument,
    :return: (float) partial fitness.
    """
    return 1 - 0.3 * x ** 2


def F_beetles(x):
    """
    Mathematical transfer function for "beetles killed" criterion.
    :param x: (float) function argument.
    :return: (float) partial fitness.
    """
    a = 0.3
    # return a ** 2 / ((1 - x) ** 2 + a ** 2)
    return x ** 2


def F_drones(x):
    """
    Mathematical transfer function for "drones died" criterion.
    :param x: (float) function argument.
    :return: (float) partial fitness.
    """
    a = 0.08
    return a ** 2 / (x ** 2 + a ** 2)


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
    Class holding all the functions and parameters for a single simulation instance.
    """

    def __init__(self, params, seed=42):
        self.score = 0  # Initialisation of fitness score for this particular simulation
        self.t = 0  # Initialisation of time [s]

        # Lists holding simulated entities
        self.entities = []
        self.trees = []
        self.drones = []
        self.beetles = []

        self.n0_drones = N_DRONES
        self.n0_beetles = N_BEETLES

        self.params = params  # tunable parameters chosen for this particular simulation to be evaluated
        self.seed = seed  # seed for random number generator

        np.random.seed(self.seed)
        self.load_environment()

        if VISUALISE:
            self.visuals = Visuals(WIDTH, HEIGHT, self.n0_drones)

    def load_environment(self):
        """
        loads simulated environment by placing trees, beetles and drones.
        :return: None
        """
        # Initial random placement of trees on map
        for i in range(int(round(N_TREES * noise(NOISE), 0))):
            placing = True
            while placing:
                x = np.random.random() * (WIDTH - 2 * TREE_MIN_DIST) + TREE_MIN_DIST // 1
                y = np.random.random() * (HEIGHT - 2 * TREE_MIN_DIST) + TREE_MIN_DIST // 1

                newtree = Entity('Tree ' + str(i), 'tree', x, y, round(np.random.normal(R_TREE_AVG, R_TREE_STD), 0))
                if not any([check_collision(newtree, entity, TREE_MIN_DIST * noise(NOISE)) for entity in self.entities]):
                    self.entities.append(newtree)
                    self.trees.append(newtree)
                    # print(newtree.name, 'placed!')
                    placing = False


        # Initial random placement of beetles on map
        for j in range(int(round(N_BEETLES * noise(NOISE), 0))):
            placing = True
            while placing:
                x = (np.random.random() * WIDTH) // 1
                y = (np.random.random() * HEIGHT) // 1

                newbeetle = Beetle('Bug ' + str(j), x, y)
                if not any([check_collision(newbeetle, entity) for entity in self.entities]):
                    self.entities.append(newbeetle)
                    self.beetles.append(newbeetle)
                    # print(newbeetle.name, 'placed!')
                    placing = False
            self.n0_beetles = len(self.beetles)

        # Initial random placement of drones on launchpad (fraction of total map)
        for k in range(int(round(N_DRONES * noise(NOISE), 0))):
            placing = True
            while placing:
                x = (np.random.random() * WIDTH * LAUNCHPAD_FRAC * noise(NOISE)) // 1
                y = (np.random.random() * HEIGHT * LAUNCHPAD_FRAC * noise(NOISE)) // 1

                newdrone = Drone('Drone ' + str(k), 'drone', x, y, self.params)
                if not any([check_collision(newdrone, entity, DRONE_MIN_DIST * noise(NOISE)) for entity in self.entities]):
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
        F1 = F_beetles(1 - len(self.beetles) / self.n0_beetles)
        F2 = F_time(self.t / T_MAX)
        F3 = F_drones(1 - len(self.drones) / self.n0_drones)
        score = F1 * F2 * F3

        return [score, (1 - len(self.drones) / self.n0_drones), (1 - len(self.beetles) / self.n0_beetles), (self.t / T_MAX)]

    def run(self):
        """
        loads environment, starts simulation loop and finally calls evaluation function.
        :return: (float) score for this particular simulation, lies in interval [0, 1].
        """
        running = True
        print(self.n0_drones, 'drones applied to ', self.n0_beetles, 'beetles.')
        while running:
            # Print time and seed every 10s
            if int(round(self.t, 0)) % 10 == 0 and abs(int(round(self.t, 0)) - self.t) < 0.001:
                print('Seed:', self.seed, 'Time:', round(self.t, 0), 's')

            # Drone simulation
            for drone in self.drones:
                for otherdrone in self.drones:
                    if check_collision(drone, otherdrone, margin=0.2*R_DRONE):
                        if drone in self.entities:
                            self.entities.remove(drone)
                        if drone in self.drones:
                            self.drones.remove(drone)  
                        self.entities.remove(otherdrone)
                        self.drones.remove(otherdrone)

                for beetle in self.beetles:
                    if check_collision(drone, beetle):
                        self.beetles.remove(beetle)
                        self.entities.remove(beetle)

                for tree in self.trees:
                    if check_collision(drone, tree, margin=0.1*R_DRONE):
                        self.drones.remove(drone)
                        self.entities.remove(drone)


                # # Check collisions
                # for entity in self.entities:
                #     if check_collision(drone, entity):
                #         # print('Collision between ', drone.name, 'and', entity.name)
                #         if entity in self.drones:
                #             if drone in self.entities:
                #                 self.entities.remove(drone)
                #             if drone in self.drones:
                #                 self.drones.remove(drone)  # XXX
                #             self.entities.remove(entity)
                #             self.drones.remove(entity)
                #
                #         elif entity in self.beetles:
                #             self.beetles.remove(entity)
                #             self.entities.remove(entity)
                #         elif entity in self.trees:


                # Maintain list of visible entities
                for entity in self.entities:
                    if drone.sees(entity) and entity not in drone.visible_entities:
                        drone.visible_entities.append(entity)
                    if not drone.sees(entity) and entity in drone.visible_entities:
                        drone.visible_entities.remove(entity)

                drone.codrones = [otherdrone for otherdrone in self.drones if not otherdrone == drone]
                drone.advance()

                for entity in drone.visible_entities:
                    if entity not in self.entities:
                        drone.visible_entities.remove(entity)

                # Bug simulation

            # Bug simulation
            for beetle in self.beetles:
                # 1. Check if drones nearby
                for drone in self.drones:
                    if beetle.sees(drone):
                        beetle.processVisual(drone)
                # First half step
                beetle.advance(DT / 2)

                # 2. Check if landed on tree or tree nearby
                for tree in self.trees:
                    if check_collision(beetle, tree):
                        beetle.mode = 'tree'
                        beetle.tree = tree
                    if beetle.sees(tree):
                        beetle.processVisual(tree)
                # Second half step
                beetle.advance(DT / 2)

            # Update screen if requested
            if VISUALISE:
                self.visuals.update(self.trees, self.beetles, self.drones)

            # Add time step
            self.t += DT

            # End conditions: 80% of drones dead, all beetles dead or time up.
            if not self.drones or not self.beetles or self.t >= T_MAX:
                running = False
                self.score = self.evaluate()

        return self.score
