import numpy as np
from Entity import Entity
from settings import *
import torch
import FlapperModel

class Drone(Entity):
    def __init__(self, name, type, x, y, bt, r_col=R_DRONE):
        super().__init__(name, type, x, y, r_col)

        # Initialisation
        self.fruit_visible = False
        self.memory = 0.0
        self.elapsed_battery_time = 0.0
        self.swarm_matrix = torch.zeros(N_DRONES, 4)
        self.message = 0.0

        self.bt = bt

        self.vx = 0
        self.vz = 0
        self.r = 0
        self.z = 0.0
        self.heading = np.random.random() * 2 * np.pi - np.pi

    def sees(self, entity) -> bool:
        """
        determines if entity is in drone's radius of vision corresponding to the type of entity.
        :param entity: Entity that is to be checked for vision.
        :return: (Boolean) True if visible, False if not.
        """
        if entity is None:
            return False

        if entity == self:
            return False

        # Determine distance
        dx = self.x - entity.x
        dy = self.y - entity.y
        dz = self.z - entity.z
        dh = np.sqrt(dx ** 2 + dy ** 2)  # Pythagoras

        elevation = np.atan(dz / dh)
        bearing = np.atan2(dy, dx) + np.pi          
        if bearing > np.pi: bearing -= 2*np.pi
        if bearing <-np.pi: bearing += 2*np.pi
        azimuth =  bearing - self.heading 

        # if self.name == 'Drone 0':
        #     print(f"Elevation: {round(elevation * 57.3, 2)}, Azimuth: {round(azimuth * 57.3, 2)}, Bearing: {round(self.heading * 57.3, 2)}, HDist: {round(dh, 2)}, VDist: {round(dz, 2)}")

        if dh <= R_TREE_AVG and np.abs(elevation) <= CAMERA_VFOV and np.abs(azimuth) <= CAMERA_HFOV:
            return True
        else:
            return False

    def advance(self):
        """
        Function executing drone dynamics.
        :return: None
        """
        cpsi = np.cos(self.heading)
        spsi = np.sin(self.heading)
        R = np.array([
            [cpsi, -spsi, 0],
            [spsi,  cpsi, 0],
            [0,        0, 1]
        ])

        for i in range(N_DRONES-1):
            if len(self.codrones) > i:
                relpos = np.array([self.codrones[i].x - self.x, self.codrones[i].y - self.y, self.codrones[i].z - self.z]) @ R
                self.swarm_matrix[i, :] = torch.Tensor([relpos[0], relpos[1], relpos[2], self.codrones[i].message])
            else: self.swarm_matrix[i, :] = torch.zeros(1, 4)
        self.swarm_matrix[N_DRONES-1, :] = torch.Tensor([self.x, self.y, self.z, self.message])

        #print(self.swarm_matrix)

        if not (MANUAL and self.name == 'Drone 0'):
            vx_cmd, vz_cmd, r_cmd, self.message = self.bt.swarm_net.forward(torch.flatten(self.swarm_matrix))
            #print(f"Action determined by SwarmNet: {self.vx}, {self.vz}, {self.r}")

        model_state = FlapperModel.advance(vx_cmd, vz_cmd, r_cmd, DT)
        y = FlapperModel.to_output(model_state)

        self.vx = y[0]
        self.vz = y[1]
        self.r = y[4]

        # Integration
        self.heading += self.r * DT
        if self.heading > np.pi: self.heading -= 2*np.pi
        if self.heading <-np.pi: self.heading += 2*np.pi


        # Transform from body to absolute frame
        self.x = self.x + self.vx * DT * np.cos(self.heading)
        self.y = self.y + self.vx * DT * np.sin(self.heading)
        self.z = self.z + self.vz * DT


    def inspect(self, fruit):
        self.fruit_visible = True


#########################################################

