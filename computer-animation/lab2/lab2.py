from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import numpy as np
from pyglet import *
from PIL import Image
from collections import deque

PARTICLE_SIZE   = 3
SPEED_CONSTANT  = 0.5
COLOR_F_R       = lambda x: x
COLOR_F_G       = lambda x: -(1/(x-2))**7
COLOR_F_B       = lambda x: (1-x)**2
ALPHA_F         = lambda x: -(x + 0.4)**(-15) + 1.2

COLOR_DICT_R    = {k:COLOR_F_R(k) for k in [round(i*0.001, 3) for i in range(0, 1001)]}
COLOR_DICT_G    = {k:COLOR_F_G(k) for k in [round(i*0.001, 3) for i in range(0, 1001)]}
COLOR_DICT_B    = {k:COLOR_F_B(k) for k in [round(i*0.001, 3) for i in range(0, 1001)]}
ALPHA_DICT      = {k:ALPHA_F(k)   for k in [round(i*0.001, 3) for i in range(0, 1001)]}

class Emitter():
    def __init__(self):
        self.particles = deque()
        
    def emit_particles(self, n=3):
        for _ in range(n): 
            p = Particle()
            self.particles.append(p)

    
    def age_particles(self):
        for p in self.particles:
            p.change_age()
        
    def randomize_particles(self):
        particles2 = deque()
        for p in self.particles: 
            if (p.age < p.lifetime):
                p.randomize()
                particles2.append(p)
        self.particles = particles2

class Particle():
    def __init__(self):
        self.pos = np.array([[0.0, -10.0, 0.0]])
        self.v = np.array([0.0, 0.0, 0.0])
        self.age = 0
        self.r = PARTICLE_SIZE
        self.c = np.array([0.0, 0.0, 0.0, 0.0])
        self.lifetime = max(1, np.random.normal(loc=100, scale=40))
    
    def change_age(self):
        self.age += 1

    def randomize(self):
        rand_vec = np.array([np.random.normal(0, 12), 6.0, 0.0])
        rand_vec = rand_vec / np.linalg.norm(rand_vec)

        t = ((self.lifetime - self.age)/self.lifetime)
        self.v = rand_vec * SPEED_CONSTANT * t**(-0.6)
        self.pos = self.pos + self.v
        self.r = (t * PARTICLE_SIZE)**2
        
        self.change_color(t)

    def change_color(self, t):
        t_round = round(t, 3)
        self.c[0] = COLOR_DICT_R[t_round]
        self.c[1] = COLOR_DICT_G[t_round]
        self.c[2] = COLOR_DICT_B[t_round]
        self.c[3] = ALPHA_DICT[t_round]

def read_texture(filename):
    w, h = 256, 256
    image = Image.open(filename)
    image = np.array(list(image.getdata()), np.uint8)
    texture = glGenTextures(1)
    gluBuild2DMipmaps(GL_TEXTURE_2D, 1, w, h, GL_RGB, GL_UNSIGNED_BYTE, image)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE)
    glEnable(GL_BLEND)
    glEnable(GL_TEXTURE_2D)
    return texture

def draw(emitter):
    glPushMatrix()
    glBegin(GL_QUADS)
    for p in emitter.particles:
        glColor4d(*(p.c))
        glTexCoord2d(0, 0); glVertex3f(*(p.pos[0] + np.array([-p.r, -p.r, 0])))
        glTexCoord2d(1, 0); glVertex3f(*(p.pos[0] + np.array([ p.r, -p.r, 0])))
        glTexCoord2d(1, 1); glVertex3f(*(p.pos[0] + np.array([ p.r,  p.r, 0])))
        glTexCoord2d(0, 1); glVertex3f(*(p.pos[0] + np.array([-p.r,  p.r, 0])))
    glEnd()
    glPopMatrix()

def set_window(width, height):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF|OPENGL)
    gluPerspective(45, width/height, 0.1, 500)
    glTranslatef(0, 0, -50)

def main():
    filename = 'cestica.bmp'
    
    width, height = 800, 600
    set_window(width, height)

    read_texture('./' + filename)
    emitter = Emitter()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # pokreni proces za stvaranje novih
        emitter.emit_particles()

        # odredi nove polozaje i druge parametre
        emitter.randomize_particles()

        # transf. sustav cestica u sustav scene na novi polozaj i poligone s texturom, iscrtavanje
        draw(emitter)

        # povecaj starost svih cestica
        emitter.age_particles()
        
        pygame.display.flip()
        pygame.time.wait(10)

main()