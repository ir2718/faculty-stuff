from OpenGL.GL import *
from OpenGL.GLU import *
from scipy import optimize
from pyglet import *
from pygame.locals import *
from sympy import Symbol
import pygame
import numpy as np
import sys
import argparse
import sympy
import colour

BOOTH_F_STR        = '(x + 2*y - 7)**2 + (2*x + y - 5)**2'
MATYAS_F_STR       = '0.26*(x**2 + y**2) - 0.48*x*y'
TEST_F_STR         = '(x - 3.14)**2 + (y - 2.72)**2 + sin(3*x + 1.41) + sin(4*y - 1.73)'
TEST_F2_STR        = 'x**2 + (y + 1)**2 - 5*cos(1.5*x + 1.5) - 3*cos(2*x -1.5)'
TEST_F3_STR        = '1/(sin(x*y)+2)'
CIRCLE_F_STR       = '(x**2 + y**2)**0.5'
CIRCLE2_F_STR      = '((x-10)**2 + (y-10)**2)**0.5'

class PSOSwarm():
    def __init__(self, pop_size, b_lo, b_up, w, fi_p, fi_g, f_str, speed_div, speed_iter, max_iter, trace_len):
        self.w = w
        self.fi_p = fi_p
        self.fi_g = fi_g
        self.f = sympy.lambdify([Symbol('x'), Symbol('y')], f_str, 'numpy')
        self.particles = [PSOParticle(b_lo, b_up) for i in range(pop_size)]
        self.g = self.find_best_pos()
        self.speed_iter = speed_iter
        self.speed_div = speed_div
        self.iter = 0
        self.max_iter = max_iter
        self.trace_len = trace_len

    def find_best_pos(self):
        best_pos = self.particles[0].get_x_and_y()
        min_val = self.f(best_pos[0], best_pos[1])
        
        for p in self.particles[1:]:
            x, y = p.get_x_and_y()
            if self.f(x, y) < min_val:
                best_pos = p.get_x_and_y()
                min_val = self.f(best_pos[0], best_pos[1])

        return np.array([best_pos[0], best_pos[1], 0])
 
    def alg_iteration(self):
        for p in self.particles:

            for j in range(len(p.v)):
                r_p = np.random.uniform(0, 1)
                r_g = np.random.uniform(0, 1)
                p.v[j] = self.w*p.v[j] + self.fi_p * r_p * (p.best_pos[j] - p.pos[j]) + self.fi_g * r_g * (self.g[j] - p.pos[j])

            p.pos = p.pos + p.v * self.speed_iter

            if self.f(p.pos[0], p.pos[1]) < self.f(p.best_pos[0], p.best_pos[1]):
                p.best_pos = p.pos
                if self.f(p.best_pos[0], p.best_pos[1]) < self.f(self.g[0], self.g[1]):
                    self.g = p.best_pos

            p.pts.append(p.pos[:])

    def calculate_curve(self):
        b =  1/6 * np.array([
            [-1, 3, -3, 1],
            [3, -6, 3, 0],
            [-3, 0, 3, 0],
            [1, 4, 1, 0],
        ])
        
        for p in self.particles:
            coords_for_all = []
            
            for i in range(len(p.pts) - 3):
                r = np.array(p.pts[i : i + 4])

                t_upper, t, step = 1, 0, 0.01
                move_step = np.round(1/self.speed_div, decimals=2)

                while t < t_upper:

                    p_vector = np.array([[t**3, t**2, t, 1]])
                    p_value = (p_vector.dot(b)).dot(r).reshape(-1)
                    coords_for_all.append(p_value)
                    
                    tmp = t % move_step
                    if  np.isclose(tmp, 0):
                        p.coordinates.append(p_value)
                        p.trace.append(coords_for_all[-self.trace_len:])
                        
                    t += step


class PSOParticle():
    def __init__(self, b_lo, b_up):
        self.pos = np.array([
            np.random.uniform(b_lo, b_up), 
            np.random.uniform(b_lo, b_up), 
            0.0
        ])
        self.best_pos = self.pos[:]
        self.v = np.array([
            np.random.uniform(-np.abs(b_up - b_lo), np.abs(b_up - b_lo)),
            np.random.uniform(-np.abs(b_up - b_lo), np.abs(b_up - b_lo)),
            0.0
        ])
        self.pts = [self.pos[:]]
        self.trace = []
        self.coordinates = []

    def get_x_and_y(self):
        return self.pos[:2]

def find_minimum_point(f_str):
    x, y = Symbol('x'), Symbol('y')
    f = sympy.parse_expr(s=f_str, local_dict={'x':x, 'y':y}, evaluate=False)

    x_l_symbols = [(x, Symbol('x[0]')), (y, Symbol('x[1]'))]
    minimum = optimize.minimize(
        sympy.lambdify([x], f.subs(x_l_symbols), 'numpy'), 
        x0=[0, 0], 
        bounds=[(-np.Inf, np.Inf), (-np.Inf, np.Inf)],
    )
    return minimum

def load_background(f, b_lo, b_up, minima, num_c, diff, rtol, atol, iso_prec, minima_x):
    grid_min = grid_max = max(abs(minima_x[0][0] - b_lo), abs(minima_x[0][1] - b_up))
    X, Y = np.meshgrid(
        np.linspace(-grid_min+minima_x[0][0], grid_max+minima_x[0][1], int(abs(grid_max+grid_min))*iso_prec), 
        np.linspace(-grid_min+minima_x[0][0], grid_max+minima_x[0][1], int(abs(grid_max+grid_min))*iso_prec)
    )

    f = sympy.lambdify([Symbol('x'), Symbol('y')], f, 'numpy')
    Z = f(X, Y)
    
    c = []
    for i in range(num_c):
        b_arr = np.isclose(Z, np.full(shape=Z.shape, fill_value=minima+i*diff), rtol=rtol, atol=atol)
        c.append(np.stack((X[b_arr], Y[b_arr]), axis=1))
    
    return c

def error_is_big_enough(swarm, err, min_):
    return np.sum([(p.get_x_and_y() - min_.reshape(-1))**2 for p in swarm.particles]) > err

def calculate_mean_of_pts(swarm):
    return np.mean([p.pos for p in swarm.particles])

def set_window(width, height, minima, b_up, b_lo):
    pygame.init()
    pygame.display.set_mode((width, height), DOUBLEBUF|OPENGL)
    gluPerspective(45, width/height, 0.1, 500)
    glTranslatef(-minima[0][0], -minima[0][1], -(b_up - b_lo + 10))
    # glTranslatef(0, 0, -50)
    glEnable(GL_POINT_SMOOTH)

def draw_background(contour, iso_colors):
    glPushMatrix()
    glPointSize(1)
    glBegin(GL_POINTS)
    for i, c in enumerate(contour):
        glColor3d(*(iso_colors[i]))
        for p in zip(c[:, 0], c[:, 1]):
            glVertex3f(*([p[0], p[1], 0]))
    glEnd()
    glPopMatrix()

def draw_swarm(swarm, t, trace_len, p_size):
    glPushMatrix()
    glColor3d(*([1, 0, 0]))
    for _, p in enumerate(swarm.particles):
        glPointSize(p_size)
        glBegin(GL_POINTS)
        glVertex3f(*(p.coordinates[t]))
        glEnd()
        
        for j, v in enumerate(p.trace[t]):
            glPointSize((j+1)/trace_len * p_size)
            glBegin(GL_POINTS)
            glVertex3f(*(v))
            glEnd()
        
    glPopMatrix()

def parse_input():
    args = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', default=CIRCLE_F_STR) 
    parser.add_argument('--pop_size', default=15, type=int) 
    parser.add_argument('--w', default=0.5, type=float)  
    parser.add_argument('--fi_p', default=2, type=float) 
    parser.add_argument('--fi_g', default=2, type=float) 
    parser.add_argument('--b_lo', default=-20, type=float)
    parser.add_argument('--b_up', default=20, type=float)

    parser.add_argument('--err', default=10**(-2), type=float) 
    parser.add_argument('--max_iter', default=10**4, type=int) 

    parser.add_argument('--speed_div', default=10, type=float) 
    parser.add_argument('--speed_iter', default=0.5, type=float) 
    parser.add_argument('--p_size', default=5, type=int) 
    parser.add_argument('--trace_len', default=20, type=int)
    parser.add_argument('--num_c', default=10, type=int) 
    parser.add_argument('--diff', default=3, type=int)
    parser.add_argument('--rtol', default=5e-4, type=float) 
    parser.add_argument('--atol', default=5e-2, type=float) 
    parser.add_argument('--iso_prec', default=15, type=int)  

    parser.add_argument('--iso_c_start', default='blue', type=str)
    parser.add_argument('--iso_c_finish', default='green', type=str)

    parser.add_argument('--width', default=800, type=int)
    parser.add_argument('--height', default=800, type=int)

    return parser.parse_args(args)

def main():
    args = parse_input()
    swarm = PSOSwarm(
        pop_size = args.pop_size, 
        b_lo = args.b_lo, 
        b_up= args.b_up, 
        w = args.w, 
        fi_p = args.fi_p, 
        fi_g = args.fi_g,
        f_str=args.f,
        speed_div = args.speed_div,
        speed_iter = args.speed_iter,
        max_iter = args.max_iter,
        trace_len =  args.trace_len
    )

    minima = find_minimum_point(args.f)
    minima_x = minima.x.reshape(1, -1)

    i = 0
    while error_is_big_enough(swarm, args.err, minima_x) and i < args.max_iter:
            swarm.alg_iteration()
            i += 1

    swarm.calculate_curve()

    set_window(args.width, args.height, minima_x, args.b_up, args.b_lo)

    c = load_background(
        args.f, 
        b_lo = args.b_lo, 
        b_up = args.b_up, 
        minima = minima.fun, 
        num_c = args.num_c,
        rtol = args.rtol,
        atol = args.atol,
        diff = args.diff,
        iso_prec = args.iso_prec,
        minima_x = minima_x
    )

    iso_colors = [c.rgb for c in list(colour.Color(args.iso_c_start).range_to(colour.Color(args.iso_c_finish), args.num_c))]

    while True:
        for t in range(len(swarm.particles[0].coordinates)):
        
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

            draw_background(c, iso_colors)
            draw_swarm(swarm, t, args.trace_len, args.p_size)

            pygame.display.flip()
            pygame.time.wait(5)

main()