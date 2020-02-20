#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified from https://github.com/idgmatrix/pygame-physics/blob/master/verlet_cloth_system_mouse2.py
"""

import pygame

from verlet_test import Cloth

pygame.init()

FPS = 60 # frames per second setting
fpsClock = pygame.time.Clock()

# set up the window
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
pygame.display.set_caption('Verlet Simple Cloth System')

# mouse button integers
MOUSELEFT = 1
MOUSERIGHT = 3

BLACK = (0, 0, 0)
            
delta_t = 0.1
# increase for more accuracy
NUM_ITER = 3

# cloth variables
NUM_X = 20
NUM_Y = 20
PARTICLE_SPACING = 20.0

Running = True
left = False
right = False
cloth = Cloth(NUM_X, NUM_Y, PARTICLE_SPACING)
while Running:
    screen.fill(BLACK)

    # particles update
    for particle in cloth.particles:
        particle.update(delta_t)

    # constraints update
    for i in range(NUM_ITER):
        for constraint in cloth.constraints:
            constraint.update()

    cloth.draw(screen)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == MOUSELEFT:
            cloth.select_particle(pygame.mouse.get_pos())
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == MOUSELEFT:
                # cloth.select_particle(pygame.mouse.get_pos())
                cloth.deselect_particle()
            elif event.button == MOUSERIGHT:
                cloth.toggle_fixed(pygame.mouse.get_pos())
            
    pygame.display.update()
    fpsClock.tick(FPS)

pygame.quit()