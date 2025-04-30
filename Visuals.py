import numpy as np
import pygame
from settings import *

def px(x, y=0):
    """helper function to transform meters to pixels. Works for points or single dimensions"""
    if y == 0:
        return int(x * SCALE)
    else:
        return (int(x * SCALE), int(y * SCALE))

class Visuals:
    """
    Class taking care of all the rendering with pygame. Here, pixels are used!
    """
    def __init__(self, width, height):
        self.FPS = 1/DT  # Determine FPS from timestep setting
        pygame.init()

        self.screen_width = px(width)
        self.screen_height = px(height)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flapper Greenhouse Swarm Simulation")

        self.font = pygame.freetype.Font(None, 12)
        self.clock = pygame.time.Clock()


        self.screen.fill(GREEN)
        pygame.display.flip()


    def update(self, trees, fruit_x_array, fruit_y_array, fruit_t_array, x_array, y_array, z_array, heading_array, active_array, t, tick):
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
                    if VIEW > 3:
                        VIEW = 0
                
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()

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

        text_surface, text_rect = self.font.render('Dead Drones: ' + str(round(N_DRONES - sum(active_array))), (0, 0, 0))
        text_rect.center = (180, 17)
        self.screen.blit(text_surface, text_rect)

        text_surface, text_rect = self.font.render('Time [s]: ' + str(round(t, 1)), (0, 0, 0))
        text_rect.center = (300, 17)
        self.screen.blit(text_surface, text_rect)

        # Draw all trees
        for tree in trees:
            pygame.draw.circle(self.screen, GREEN, px(tree.x, tree.y), px(tree.r_col))

        for j in range(N_FRUIT):
            pygame.draw.circle(self.screen, RED, px(fruit_x_array[j], fruit_y_array[j]), px(R_FRUIT))
            pygame.draw.circle(self.screen, RED, px(fruit_x_array[j], fruit_y_array[j]), px(R_TREE_AVG), 1)
            text_surface, text_rect = self.font.render((str(round(fruit_t_array[j, tick], 1))), (0, 0, 0))
            text_rect.center = px(fruit_x_array[j], fruit_y_array[j])
            self.screen.blit(text_surface, text_rect)


        for i in range(N_DRONES):
            if active_array[i]:
                pygame.draw.circle(self.screen, BLUE, px(x_array[i], y_array[i]), px(R_DRONE))
                pygame.draw.line(self.screen, RED, px(x_array[i], y_array[i]), (float(px(x_array[i]) + np.cos(heading_array[i]) * px(R_DRONE)), float(px(y_array[i]) + np.sin(heading_array[i]) * px(R_DRONE))), 2)
                
                # Height indication
                text_surface, text_rect = self.font.render(str(round(z_array[i], 2)), (255, 255, 255))
                text_rect.center = px(x_array[i], y_array[i])
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


        # Screen update
        pygame.display.flip()


        # Wait until frame time is up to create real-time impression
        if REALTIME:
            self.clock.tick(self.FPS)


