#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified from https://github.com/idgmatrix/pygame-physics/blob/master/verlet_cloth_system_mouse2.py
"""

import math

import pygame

# # set up the window
WIDTH = 800
HEIGHT = 600

PARTICLE_SIZE = 3
LINE_THICKNESS = 1

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 255, 255)

class Particle:
    def __init__(self, x, y, m = 1.0):
        self.m = m
        self.x = x
        self.y = y
        self.oldx = x
        self.oldy = y
        self.ax = 0
        self.ay = 9.8
        
        self.fixed = False
        self.selected = False
        
    def update(self, dt):
        if not self.fixed:
            # Verlet Integration
            newx = 2.0 * self.x - self.oldx + self.ax * dt * dt
            newy = 2.0 * self.y - self.oldy + self.ay * dt * dt
            # newx = self.x + (self.x - self.oldx) * 0.99 + self.ax * dt * dt
            # newy = self.y + (self.y - self.oldy) * 0.99 + self.ay * dt * dt
            self.oldx = self.x
            self.oldy = self.y
            self.x = newx
            self.y = newy
            
            # Collision Process
            if self.x < 0 or self.x > WIDTH:
                self.x, self.oldx = self.oldx, self.x
            if self.y < 0 or self.y > HEIGHT:
                self.y, self.oldy = self.oldy, self.y
                
            # if self.x > WIDTH:
            #     self.x = WIDTH
            #     self.oldx = self.x + vx
            # elif self.x < 0:
            #     self.x = 0
            #     self.oldx = self.x + vx
            # if self.y > HEIGHT:
            #     self.y = HEIGHT
            #     self.oldy = self.y + vy
            # elif self.y < 0:
            #     self.y = 0
            #     self.oldy = self.y + vy
            
        if self.selected == True:
            self.x, self.y = pygame.mouse.get_pos()
        
    def draw(self, surf, size):
        if self.selected:
            color = RED
        elif self.fixed:
            color = BLUE
        else:
            color = WHITE        
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), size)

       
class Constraint:
    def __init__(self, particle1, particle2):
        self.particle1 = particle1
        self.particle2 = particle2
        dx = particle1.x - particle2.x
        dy = particle1.y - particle2.y
        self.restLength = math.sqrt(dx*dx + dy*dy)
        
    def update(self):
        dx = self.particle1.x - self.particle2.x
        dy = self.particle1.y - self.particle2.y
        deltaLength = math.sqrt(dx*dx + dy*dy)
        # diff = (deltaLength - self.restLength) / (deltaLength + 0.001)
        diff = (self.restLength - deltaLength) / deltaLength

        if not self.particle1.fixed:
            self.particle1.x += 0.5 * diff * dx
            self.particle1.y += 0.5 * diff * dy
        if not self.particle2.fixed:
            self.particle2.x -= 0.5 * diff * dx
            self.particle2.y -= 0.5 * diff * dy
            
    def draw(self, surf, size):
        x0 = self.particle1.x
        y0 = self.particle1.y
        x1 = self.particle2.x
        y1 = self.particle2.y
        pygame.draw.line(surf, WHITE, (int(x0), int(y0)), (int(x1), int(y1)), size)
        

class Cloth:
    def __init__(self, numX, numY, particleSpacing):
        self.particles = []
        for j in range(numY):
            for i in range(numX):
                x = 100 + i * particleSpacing
                y = 50 + j * particleSpacing
                p = Particle(x, y)
                self.particles.append(p)

        self.particles[0].fixed = True
        self.particles[numX-1].fixed = True
        self.particles[(numY-1) * numX].fixed = True
        self.particles[(numY) * numX - 1].fixed = True

        self.constraints = []
        for j in range(numY):
            for i in range(numX):
                if i < (numX - 1):
                    particle1 = self.particles[i + j * numX]
                    particle2 = self.particles[(i + 1) + j * numX]
                    c = Constraint(particle1, particle2)
                    self.constraints.append(c)
                if j < (numY - 1):
                    particle1 = self.particles[i + j * numX]
                    particle2 = self.particles[i + (j + 1) * numX]
                    c = Constraint(particle1, particle2)
                    self.constraints.append(c)

        # for j in range(numY - 1):
        #     for i in range(numX - 1):
        #         particle1 = self.particles[i + j * numX]
        #         particle2 = self.particles[(i + 1) + (j + 1) * numX]
        #         c = Constraint(particle1, particle2)
        #         self.constraints.append(c)
        #     for i in range(1, numX):
        #         particle1 = self.particles[i + j * numX]
        #         particle2 = self.particles[(i - 1) + (j + 1) * numX]
        #         c = Constraint(particle1, particle2)
        #         self.constraints.append(c)


    # TODO: figure out how to select one only when overlapping
    def select_particle(self, pos):
        for particle in self.particles:
            # if particle.selected:
            #     particle.selected = False
            #     break
            dx = particle.x - pos[0]
            dy = particle.y - pos[1]
            if math.sqrt(dx*dx + dy*dy) < PARTICLE_SIZE + 10:
                particle.selected = True
                break

    def deselect_particle(self):
        for particle in self.particles:
            if particle.selected:
                particle.selected = False
                break

    def toggle_fixed(self, pos):
        for particle in self.particles:
            dx = particle.x - pos[0]
            dy = particle.y - pos[1]
            if math.sqrt(dx*dx + dy*dy) < PARTICLE_SIZE + 5:
                particle.fixed = not particle.fixed
                break

    def draw(self, surf):
        # particles draw
        for particle in self.particles:
            particle.draw(surf, 3)
        # constraints draw
        for constraint in self.constraints:
            constraint.draw(surf, 1)