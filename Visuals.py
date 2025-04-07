import numpy as np
import pygame
from settings import *


class Visuals:
    def __init__(self, width, height, n0_drones):
        self.FPS = 1/DT  # Determine FPS from timestep setting
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Bug Catching Swarming Simulation")

        self.font = pygame.freetype.Font(None, 12)
        self.clock = pygame.time.Clock()

        self.n0_drones = n0_drones

        self.screen.fill(GREEN)
        pygame.display.flip()


    def update(self, trees, drones):
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

        self.screen.fill(GREEN)

        # Simulation information texts
        text_surface, text_rect = self.font.render('Active Drones: ' + str(round(len(drones))), (0, 0, 0))
        text_rect.center = (60, 15)
        self.screen.blit(text_surface, text_rect)

        text_surface, text_rect = self.font.render('Dead Drones: ' + str(round(self.n0_drones - len(drones))), (0, 0, 0))
        text_rect.center = (180, 15)
        self.screen.blit(text_surface, text_rect)

        # Draw all trees
        for tree in trees:
            pygame.draw.circle(self.screen, BROWN, (tree.x, tree.y), tree.r_col)


        for drone in drones:
            pygame.draw.circle(self.screen, GREY, (drone.x, drone.y), drone.r_col)

            if VIEW == 1:  # Drone vision and influenced entitites
                pygame.draw.circle(self.screen, GREY, (drone.x, drone.y), drone.r_vis['drone'], 1)  # visual range for drones
                pygame.draw.circle(self.screen, RED, (drone.x, drone.y), drone.r_vis['beetle'], 1)  # visual range for beetles
                for entity in drone.visible_entities:
                    pygame.draw.line(self.screen, TYPE_COLOURS[entity.type], (drone.x, drone.y), (entity.x, entity.y), 1)

            if VIEW == 2:  # Mid-range inter-drone communication, activity labels
                pygame.draw.circle(self.screen, WHITE, (drone.x, drone.y), drone.r_fardrone, 1)
                pygame.draw.circle(self.screen, WHITE, (drone.x, drone.y), drone.r_activity, 2)

                text_surface, text_rect = self.font.render(str(round(drone.activity, 0)), (0, 0, 0))
                text_rect.center = (drone.x, drone.y)
                self.screen.blit(text_surface, text_rect)


        # Screen update
        pygame.display.flip()


        # Wait until frame time is up to create real-time impression
        if REALTIME:
            self.clock.tick(self.FPS)


