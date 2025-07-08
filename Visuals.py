import numpy as np
import pygame
from settings import *
from numba import njit

_half_extent = R_TOF * np.tan(TOF_HFOV / 2)
_apex = np.array([0.0, 0.0, 0.0])
_base1 = np.array([R_TOF, -_half_extent, -_half_extent])
_base2 = np.array([R_TOF,  _half_extent, -_half_extent])
_base3 = np.array([R_TOF, -_half_extent,  _half_extent])
_base4 = np.array([R_TOF,  _half_extent,  _half_extent])



def _px(x, y=None):
    """helper function to transform meters to pixels. Works for points or single dimensions"""
    if y is None:
        return int(x * SCALE)
    else:
        return (int(x * SCALE), int(y * SCALE))

@njit
def _fcolour(disc):
    if disc: return BLACK
    else: return RED

class Visuals:
    """
    Class taking care of all the rendering with pygame. Here, pixels are used!
    """
    def __init__(self, width, height, n_rows):
        self.FPS = 1/DT  # Determine FPS from timestep setting
        pygame.init()

        self.screen_width = _px(width)
        self.screen_height = _px(height)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flapper Greenhouse Swarm Simulation")

        self.font = pygame.freetype.Font(None, 12)
        self.clock = pygame.time.Clock()
        self.n_rows = n_rows


        self.screen.fill(GREEN)
        pygame.display.flip()


    def draw_fov_triangle(self, tip, fov, height, theta, color=(0, 255, 255)):

        height = height / np.cos(fov) 
        rx = int(tip[0] + height * np.cos(theta + fov/2))
        ry = int(tip[1] + height * np.sin(theta + fov/2))

        lx = int(tip[0] + height * np.cos(theta - fov/2))
        ly = int(tip[1] + height * np.sin(theta - fov/2))

        points = [tip, (rx, ry), (lx, ly)]

        pygame.draw.polygon(self.screen, color, points)


    def update(self, trees,  x_array, y_array, z_array, heading_array, active_array, string_array, t, tick):
        # Make sure visualisation is ended when window is closed
        global VIEW

        # Event Listener
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            # Key Listener
            elif event.type == pygame.KEYDOWN:

                # Change view
                if event.key == pygame.K_v:
                    VIEW += 1
                    print(f'View changed to {VIEW}')
                    if VIEW > 1:
                        VIEW = 0
                
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()

                if event.key == pygame.K_RETURN:
                    return True

                # Pause simulation
                elif event.key == pygame.K_SPACE:
                    pause = True
                    while pause:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    pause = False

        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, GREY, pygame.Rect(BORDERSIZE, BORDERSIZE, self.screen_width - 2*BORDERSIZE, self.screen_height - 2*BORDERSIZE))

        #Simulation information texts
        text_surface, text_rect = self.font.render('Active Drones: ' + str(sum(active_array)), (0, 0, 0))
        text_rect.center = (60, 17)
        self.screen.blit(text_surface, text_rect)

        text_surface, text_rect = self.font.render('Crashed Drones: ' + str(round(N_DRONES - sum(active_array))), (0, 0, 0))
        text_rect.center = (180, 17)
        self.screen.blit(text_surface, text_rect)

        text_surface, text_rect = self.font.render('Time [s]: ' + str(round(t, 1)), (0, 0, 0))
        text_rect.center = (300, 17)
        self.screen.blit(text_surface, text_rect)

        # text_surface, text_rect = self.font.render('Discovered Fruit: ' + str(round(sum(fruit_disc_array))), (0, 0, 0))
        # text_rect.center = (480, 17)
        # self.screen.blit(text_surface, text_rect)

        # Draw all trees
        # for tree in trees:
        #     pygame.draw.circle(self.screen, GREEN, px(tree.x, tree.y), px(tree.r_col))
        
        for row in range(self.n_rows):
            pygame.draw.rect(self.screen, BLUE, pygame.Rect(
                _px(LAUNCHPAD_FRAC*WIDTH + R_TREE_AVG),
                _px(HEIGHT / (self.n_rows + 1) * (row+1) - 0.5*R_TREE_AVG),
                _px((1-LAUNCHPAD_FRAC)*WIDTH - 2*R_TREE_AVG), 
                _px(R_TREE_AVG))
                                                            )

        # for j in range(N_FRUIT):
        #     pygame.draw.circle(self.screen, fcolour(fruit_disc_array[j]), px(fruit_x_array[j], fruit_y_array[j]), px(R_FRUIT))
        #     #pygame.draw.circle(self.screen, fcolour(fruit_disc_array[j]), px(fruit_x_array[j], fruit_y_array[j]), px(R_DISCOVERY), 1)
        #     text_surface, text_rect = self.font.render((str(round(fruit_z_array[j], 1))), (0, 0, 0))
        #     text_rect.center = px(fruit_x_array[j], fruit_y_array[j])
        #     self.screen.blit(text_surface, text_rect)


        for i in range(N_DRONES):
            if i == 0: colour = GREEN
            else: colour = BLUE
            if active_array[i]:
                pygame.draw.circle(self.screen, colour, _px(x_array[i, tick], y_array[i, tick]), _px(R_DRONE))
                pygame.draw.line(self.screen, RED, _px(x_array[i, tick], y_array[i, tick]), (float(_px(x_array[i, tick]) + np.cos(heading_array[i]) * _px(R_DRONE)), float(_px(y_array[i, tick]) + np.sin(heading_array[i]) * _px(R_DRONE))), 2)
                self.draw_fov_triangle(_px(x_array[i, tick], y_array[i, tick]), TOF_HFOV, _px(R_TOF), heading_array[i])

                # Height indication
                text_surface, text_rect = self.font.render(str(round(z_array[i], 2)), (255, 255, 255))
                text_rect.center = _px(x_array[i, tick], y_array[i, tick])
                self.screen.blit(text_surface, text_rect)

                # state indication
                text_surface, text_rect = self.font.render(string_array[i], (0, 0, 0))
                text_rect.center = _px(x_array[i, tick]+0.25, y_array[i, tick])
                self.screen.blit(text_surface, text_rect)
                
            # if VIEW == 1:  # Drone vision and influenced entitites
            #     pygame.draw.circle(self.screen, GREY, (X, Y), drone.r_vis['drone'], 1)  # visual range for drones
            #     for entity in drone.visible_entities:
            #         pygame.draw.line(self.screen, TYPE_COLOURS[entity.type], (X, Y), (entity.x * SCALE, entity.y * SCALE), 1)
                    

            # if VIEW == 2:  # Mid-range inter-drone communication, activity labels
            #     pygame.draw.circle(self.screen, WHITE, (X, Y), drone.r_fardrone, 1)
            #     pygame.draw.circle(self.screen, WHITE, (X, Y), drone.r_activity, 2)

            #     text_surface, text_rect = self.font.render(str(round(drone.activity, 0)), (0, 0, 0))
            #     text_rect.center = (X, Y)
            #     self.screen.blit(text_surface, text_rect)
       
        if VIEW == 1:
            x_flat = x_array.ravel()
            y_flat = y_array.ravel()
            for i in range(len(x_flat)):
                #print(_px(x_flat[i], y_flat[i]))
                
                pygame.draw.circle(self.screen, RED, _px(x_flat[i], y_flat[i]), 1)

        # Screen update
        pygame.display.flip()


        # Wait until frame time is up to create real-time impression
        if REALTIME:
            self.clock.tick(self.FPS)
        
        return False


