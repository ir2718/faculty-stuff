from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import numpy as np
from obj_reader import ObjObject
from math import degrees, acos

SCALE_CONSTANT = 0.5

def draw_object(obj, translate=[0, 0, 0], rotation=([0, 0, 0], 0)):
    glPushMatrix()

    glTranslate(translate[0], translate[1], translate[2])
    glRotatef(
        rotation[1], 
        rotation[0][0],
        rotation[0][1],
        rotation[0][2]
    )
    

    glBegin(GL_TRIANGLES)
    glColor3fv((0.5, 0.5, 0.5))
    for face in obj.f:
        glVertex3fv(obj.v[face[0]-1])
        glVertex3fv(obj.v[face[1]-1])
        glVertex3fv(obj.v[face[2]-1])
    glEnd()

    glPopMatrix()

def read_coordinates():
    ret_l = []
    for l in open('./koordinate.txt'):
        nums = [float(i) for i in l.split(' ')]
        ret_l.append(np.array(nums))
    return ret_l


def calculate_animation():
    b =  1/6 * np.array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 0, 3, 0],
        [1, 4, 1, 0],
    ])

    b_der = 1/2 * np.array([
        [-1, 3, -3, 1],
        [2, -4, 2, 0],
        [-1, 0, 1, 0],
    ])

    spirals = read_coordinates()

    tangent = []
    translation = []
    rotation_plane_and_angle = []

    s = np.array([0, 0, 1])

    for i in range(len(spirals) - 3):
        r = np.array(spirals[i : i + 4])

        t_upper, t, step = 1, 0, 0.02
        while (t < t_upper):
            
            # odredi ciljnu orijentaciju objekta
            p_vector_der = np.array([[t**2, t, 1]])
            p_value_der = (p_vector_der.dot(b_der)).dot(r)
            tangent.append(p_value_der)

            # odredi putanju objekta
            p_vector = np.array([[t**3, t**2, t, 1]])
            p_value = (p_vector.dot(b)).dot(r)
            translation.append(p_value)

            # s = [x, y, z]
            # e = [x, y, z]
            #
            # sy ez - ey sz =====> s[1]*e[2] - e[1]*s[2]
            # -(sx ez - es sz) ==> -s[0]*e[2] + e[0]*s[2]
            # sx ey - sy ex =====> s[0]*e[1] - s[1]*e[0]

            e = p_value_der[0]
            plane = np.array([
                [s[1]*e[2] - e[1]*s[2]],
                [-s[0]*e[2] + e[0]*s[2]],
                [s[0]*e[1] - s[1]*e[0]]
            ]) 
            cos_fi = s.dot(p_value_der.T) / (np.linalg.norm(s) * np.linalg.norm(p_value_der))
            
            rotation_plane_and_angle.append((plane, degrees(acos(cos_fi))))

            t += step
            s = (p_value + p_value_der)[0]
    return translation, tangent, rotation_plane_and_angle

def draw_spiral(translation):
    glPushMatrix()

    glBegin(GL_LINES)
    glColor3fv((1, 0, 0))
    for i in range(len(translation) - 1):
        glVertex3fv(translation[i])
        glVertex3fv(translation[i+1])
    glEnd()
    glPopMatrix()

def draw_tangent(p1, p2):
    glPushMatrix()

    glColor3fv((0, 1, 0))
    glBegin(GL_LINES)
    glVertex3fv(p1)
    glVertex3fv(p1 + p2 * SCALE_CONSTANT)
    glEnd()
    
    glPopMatrix()

def main():
    obj = ObjObject()
    obj.parse('./models/kocka.obj')

    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, display[0]/display[1], 0.1, 500)
    
    glTranslatef(0, 0, -75.0)

    translation, tangent, rotation = calculate_animation()

    t_i = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        draw_spiral(translation)
        draw_object(obj, translation[t_i].tolist()[0], rotation[t_i])
        draw_tangent(translation[t_i], tangent[t_i])
        pygame.display.flip()
        pygame.time.wait(10)

        t_i = (t_i + 1) % len(translation)

main()