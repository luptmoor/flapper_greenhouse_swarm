import numpy as np
from Entity import Entity
from settings import *


class Drone(Entity):
    def __init__(self, name, type, x, y, bt, r_col=R_DRONE):
        super().__init__(name, type, x, y, r_col)

        # # Tunable Parameters, negative ks mean attraction, positive means repulsion
        # self.r_vis_tree = params[0] * RANGE_R_VIS_TREE / 2 + MU_R_VIS_TREE
        # self.k_tree = params[1] * RANGE_K_TREE / 2 + MU_K_TREE

        # self.r_vis_beetle = params[2] * RANGE_R_VIS_BEETLE / 2 + MU_R_VIS_BEETLE
        # self.k_beetle = params[3] * RANGE_K_BEETLE / 2 + MU_K_BEETLE

        # self.r_vis_neardrone = params[4] * RANGE_R_VIS_NEARDRONE / 2 + MU_R_VIS_NEARDRONE
        # self.k_neardrone = params[5] * RANGE_K_NEARDRONE / 2 + MU_K_NEARDRONE

        # self.r_vis = {'tree': self.r_vis_tree, 'drone': self.r_vis_neardrone, 'beetle': self.r_vis_beetle}
        # self.gains = {'tree': self.k_tree, 'drone': self.k_neardrone, 'beetle': self.k_beetle}

        # self.r_fardrone = params[6] * RANGE_R_VIS_FARDRONE / 2 + MU_R_VIS_FARDRONE
        # self.k_fardrone = params[7] * RANGE_K_FARDRONE / 2 + MU_K_FARDRONE

        # self.r_activity = params[8] * RANGE_R_ACTIVITY / 2 + MU_R_ACTIVITY
        # self.k_activity = params[9]  * RANGE_K_ACTIVITY / 2 + MU_K_ACTIVITY

        # self.v_min = min(V_DRONE_MAX, max(0, params[10] * RANGE_V_MIN / 2 + MU_V_MIN))  # clip between 0 and V_DRONE_MAX
        # self.v_max = min(V_DRONE_MAX, max(self.v_min, params[11] * RANGE_V_MAX / 2 + MU_V_MAX))  # clip between v_min and V_DRONE_MAX

        # self.c = min(1, max(params[12] * RANGE_C / 2 + MU_C, 0))  # clip between 0 and 1


        # Initialisation
        self.fruit_visible = False
        self.memory = 0.0
        self.elapsed_battery_time = 0.0

        self.bt = bt

        self.vx = 0
        self.vz = 0
        self.z = 0.0
        self.ax = 0
        self.ay = 0
        self.heading = np.random.random() * 2 * np.pi

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

        # Determine distance according to periodical domain
        dx = self.x - entity.x
        dy = self.y - entity.y
        d = np.sqrt(dx ** 2 + dy ** 2)  # Pythagoras


        obstructed = False
        # If a beetle, check if it's behind a tree
        if entity.type == 'beetle' and entity.tree is not None:
            d_obs = np.sqrt((self.x - entity.tree.x) ** 2 + (self.y - entity.tree.y) ** 2)
            # If tree is closer than beetle, beetle is probably not visible
            if d_obs < d:
                obstructed = True

        if d <= (self.r_vis[entity.type] + entity.r_col) and not obstructed:
            return True
        else:
            return False

    def advance(self):
        """
        Function executing drone dynamics.
        :return: None
        """

        blackboard = {
            "elapsed_battery_time": self.elapsed_battery_time,
            "fruit_visible": self.fruit_visible,
            "memory": self.memory
        }
        self.vx, self.vz, self.r = self.bt.feed_forward(blackboard)


        # Integration
        self.heading += self.r * DT
        if self.heading > np.pi or self.heading < -np.pi: self.heading = (self.heading + np.pi) % (2*np.pi) - np.pi

        self.x = int(round(self.x + self.vx  * DT, 0))
        self.z = int(round(self.z + self.vz  * DT, 2))


