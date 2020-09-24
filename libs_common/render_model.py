import numpy

import glfw

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *


class RenderModel:
    def __init__(self, width = 512, height = 512):
        
        self.width  = width
        self.height = height

        self._create_window()
    

    def _create_window(self):

        # Initialize the library
        if not glfw.init():
            return -1
        
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(self.width, self.height, "render model", None, None)
        if not self.window:
            glfw.terminate()
            return -2

        # Make the window's context current
        glfw.make_context_current(self.window)
        
        return 0

    
    def render(self, points, points_initial, edges, color = [1.0, 0.0, 0.0]):
        aspect = self.width/self.height
        glViewport(0, 0, 2*self.width, 2*self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        glOrtho(-1*aspect, 1*aspect, -1, 1, -1, 1)

        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glClearColor(1, 1, 1, 0)
        

        self._render(points, points_initial, edges, color)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

        return glfw.window_should_close(self.window)

    def _render(self, points, points_initial, edges, color):
        
        position         = numpy.transpose(points)
        position_initial = numpy.transpose(points_initial)

        max = numpy.max(position_initial)
        min = numpy.min(position_initial)

        position = (position - min)/(max - min)
        position = 0.5*(2.0*position - 1.0)

        glBegin(GL_LINES) 

        glColor3f(0.9*color[0], 0.9*color[1], 0.9*color[2]) 
 
        for i in range(edges.shape[1]):
            idx_a = edges[0][i]
            idx_b = edges[1][i]

            xa = position[0][idx_a]
            ya = position[1][idx_a]
            za = position[2][idx_a]

            xb = position[0][idx_b]
            yb = position[1][idx_b]
            zb = position[2][idx_b]
            
            glVertex3f(xa, ya, za)
            glVertex3f(xb, yb, zb)
        
        glEnd()

        glBegin(GL_POINTS) 

        glColor3f(color[0], color[1], color[2]) 
        
        for i in range(points.shape[1]):
            xa = position[0][i]
            ya = position[1][i] 
            za = position[2][i]

            glVertex3f(xa, ya, za)

        glEnd()


