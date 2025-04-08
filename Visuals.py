import numpy as np
import pygame
from settings import *


class Visuals:
    """
    Class taking care of all the rendering with pygame. Here, pixels are used!
    """
    def __init__(self, width, height, n0_drones):
        self.FPS = 1/DT  # Determine FPS from timestep setting
        pygame.init()

        self.screen_width = width * SCALE
        self.screen_height = height * SCALE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Flapper Greenhouse Swarm Simulation")

        self.font = pygame.freetype.Font(None, 12)
        self.clock = pygame.time.Clock()

        self.n0_drones = n0_drones

        self.screen.fill(GREEN)
        pygame.display.flip()


    def update(self, trees, fruits, drones):
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

        # Simulation information texts
        text_surface, text_rect = self.font.render('Active Drones: ' + str(round(len(drones))), (0, 0, 0))
        text_rect.center = (60, 15)
        self.screen.blit(text_surface, text_rect)

        text_surface, text_rect = self.font.render('Dead Drones: ' + str(round(self.n0_drones - len(drones))), (0, 0, 0))
        text_rect.center = (180, 15)
        self.screen.blit(text_surface, text_rect)

        # Draw all trees
        for tree in trees:
            pygame.draw.circle(self.screen, GREEN, (tree.x * SCALE, tree.y * SCALE), tree.r_col * SCALE)

        for fruit in fruits:
            pygame.draw.circle(self.screen, RED, (fruit.x * SCALE, fruit.y * SCALE), fruit.r_col * SCALE)
            text_surface, text_rect = self.font.render((str(round(fruit.record[-1], 1))), (0, 0, 0))
            text_rect.center = (fruit.x*SCALE, fruit.y*SCALE)
            self.screen.blit(text_surface, text_rect)


        for drone in drones:
            X = int(drone.x * SCALE)
            Y = int(drone.y * SCALE)

            pygame.draw.circle(self.screen, BLUE, (X, Y), R_DRONE * SCALE)
            pygame.draw.line(self.screen, RED, (X, Y), (float(X+ np.cos(drone.heading) * R_DRONE * SCALE), float(Y + np.sin(drone.heading) * R_DRONE * SCALE)), 2)
            
            # Height indication
            text_surface, text_rect = self.font.render(str(round(drone.z, 2)), (255, 255, 255))
            text_rect.center = (X, Y)
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


