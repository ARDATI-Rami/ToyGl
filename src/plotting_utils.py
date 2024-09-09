# src/plotting_utils.py

import numpy as np
import time
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from src.math_utils import normalize, norm
from src.logging_utils import logger
# Create a class to handle plotting
class Plotting:
    def __init__(self, eptm):
        self.eptm = eptm
        self.azim = 0
        self.elev = 0
        self.distance = 50
        self.origine = [0, 0, 0]
        self.mouse_orig = [0, 0]
        self.rotate = False
        self.zoom = False
        self.translation = False
        self.is_paused = False
        self.previousTime = time.time()
        self.frameCounter = 0
        self.frameCounterTimer = time.time()

        ###################################
        # Intialisation GLUT
        glutInit('')
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
        glutInitWindowSize(640, 480)
        window = glutCreateWindow('PyToyGL')
        glClearColor(1, 1, 1, 1)
        glClearDepth(1)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.1, 0.1, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glShadeModel(GL_FLAT)
        glCullFace(GL_BACK)
        glEnable(GL_CULL_FACE)
        glEnable(GL_COLOR_MATERIAL)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

        # respond to event
        glutDisplayFunc(self.display)
        # glutIdleFunc(self.animation)
        glutReshapeFunc(self.resize)
        glutIgnoreKeyRepeat(1)
        glutKeyboardFunc(self.key)
        glutMouseFunc(self.mouse)
        glutMotionFunc(self.motion)

    def draw_cylinder(self,pt1, pt2, rayon1, rayon2):
        """
        Draw a cylinder between pt1 and pt2
        """
        dpt = np.array(pt2) - np.array(pt1)
        v = norm(dpt)
        if dpt[2] == 0:
            dpt[2] = 0.0001
        ax = 57.2957795 * math.acos(dpt[2] / v)
        if dpt[2] < 0.:
            ax = -ax
        rx = -dpt[1] * dpt[2]
        ry = dpt[0] * dpt[2]
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(pt1[0], pt1[1], pt1[2])
        glRotatef(ax, rx, ry, 0.)
        quad = gluNewQuadric()
        gluQuadricOrientation(quad, GLU_OUTSIDE)
        gluQuadricDrawStyle(quad, GLU_FILL)
        gluCylinder(quad, rayon1, rayon2, v, 8, 1)
        glPopMatrix()

    def draw_sphere(self,pos, radius):
        """
        Draw a sphere at the indicated position, of the indicated radius
        (10 subdivisions by default)

        """
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(pos[0], pos[1], pos[2])
        glutSolidSphere(radius, 10, 5)
        glPopMatrix()

    def draw_arrow(self,start_point, end_point, arrow_radius=0.05, head_length=0.2, head_radius=0.1, color=(1.0, 1.0, 1.0)):
        """
        Draw an arrow from start_point to end_point.

        Args:
            start_point (tuple): The starting point of the arrow (x, y, z).
            end_point (tuple): The ending point of the arrow (x, y, z).
            arrow_radius (float): Radius of the arrow shaft.
            head_length (float): Length of the arrowhead.
            head_radius (float): Radius of the arrowhead.
            color (tuple): Color of the arrow (R, G, B).
        """
        # Calculate the vector from start to end point and its length
        vector = list(np.array(end_point) - np.array(start_point))
        length = norm(vector)

        # Normalize the vector
        direction = normalize(vector)

        # Calculate the rotation angle and axis
        rotation_axis = np.cross([0, 0, 1], direction)
        rotation_angle = np.arccos(np.dot([0, 0, 1], direction)) * 180.0 / np.pi

        # Set up the transformation and rotation
        glPushMatrix()
        glColor3f(*color)
        glTranslatef(*start_point)
        if length != 0:
            glRotatef(rotation_angle, *rotation_axis)

        # Draw the shaft of the arrow
        quad = gluNewQuadric()
        gluQuadricOrientation(quad, GLU_OUTSIDE)
        gluQuadricDrawStyle(quad, GLU_FILL)
        gluCylinder(quad, arrow_radius, arrow_radius, length - head_length, 8, 1)

        # Draw the arrowhead
        glTranslatef(0, 0, length - head_length)
        gluCylinder(quad, head_radius, 0.0, head_length, 8, 1)

        # Restore the matrix state
        glPopMatrix()

    def draw_reference_frame(self):
        self.draw_arrow(start_point=(0, 0, 0), end_point=(7, 0, 0), color=(1, 0, 0))  # X-axis (Red)
        self.draw_arrow(start_point=(0, 0, 0), end_point=(0, 7, 0), color=(0, 1, 0))  # Y-axis (Green)
        self.draw_arrow(start_point=(0, 0, 0), end_point=(0, 0, 7), color=(0, 0, 1))  # Z-axis (Blue)

    def draw_plane(self):
        """
        Drawing of the base plan

        """
        int_x = -10, 10
        int_y = -10, 10
        glBegin(GL_LINES)
        for x in range(int_x[0], int_x[1] + 1, 1):
            glVertex3f(x, int_y[0], 0.)
            glVertex3f(x, int_y[1], 0.)
        for y in range(int_y[0], int_y[1] + 1, 1):
            glVertex3f(int_x[0], y, 0.)
            glVertex3f(int_x[1], y, 0.)
        glEnd()

    def display(self):
        # preparation timing
        currentTime = time.time()
        deltaTime = (currentTime - self.previousTime)

        # effacement + placement camera
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear buffers
        glEnable(GL_DEPTH_TEST)  # Enable depth testing
        glEnable(GL_BLEND)  # Enable blending
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Set blend function

        self.setCamera()

        #########################################
        # Draw nodes, filaments, facets, reference frame, and plane
        for node in self.eptm.get_all_nodes():
            node.draw(selected=True, in_contact=False)

        for filament in self.eptm.get_all_filaments():
            filament.draw(selected=False)

        for facet in self.eptm.get_all_facets():
            facet.draw(color=(0, 0, 1, 0.5))  # blue color with 50% transparency
        if self.eptm.adhesions:
            for adhesion in self.eptm.adhesions:
                print(f"adhesion : {adhesion}")
                adhesion.draw(selected=True)
        self.draw_reference_frame()
        self.draw_plane()

        #########################################
        # basculement buffer : affichage
        glutSwapBuffers()

        # timing
        self.previousTime = currentTime
        self.frameCounter += 1
        if currentTime - self.frameCounterTimer > 1:
            # affichage toutes les secondes
            logger.info(f"FPS: {self.frameCounter}")
            self.frameCounter = 0
            self.frameCounterTimer = currentTime

    def resize(self,width, height):
        """
        Resizing the window and rotating the viewport for XZ plane
        """
        if height == 0: height = 1
        screenSize = [width, height]
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, float(width) / float(height), 0.1, 1000.0)  # Field of View set to 45 degrees
        glMatrixMode(GL_MODELVIEW)

        # Rotate the viewport 90 degrees around the X-axis to align with XZ plane
        glRotatef(270, 0.0, 0.0, 1.0)

    def setCamera(self):

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Apply transformations based on updated azim and elev values
        glTranslatef(0., 0., -self.distance)
        # Set elevation and azimuth to align with the XZ plane
        # elev = 80 # Rotate 90 degrees around the X-axis to switch from XY to XZ plane
        # azim = 0 # No rotation around the Y-axis needed for this plane
        glRotatef(self.elev, 1, 0., 0.)  # Rotate around the x-axis for elevation
        glRotatef(self.azim, 0., 1, 0.)  # Rotate around the y-axis for azimuth
        glTranslatef(-self.origine[0], -self.origine[1], -self.origine[2])

    def motion(self,x, y):
        """
        self.mouse movement management
        """

        if self.rotate:
            self.azim += 0.2 * (x - self.mouse_orig[0])
            self.elev += 0.2 * (y - self.mouse_orig[1])
            self.azim = self.azim % 360.0
            self.elev = self.elev % 360.0
            self.mouse_orig[0] = x
            self.mouse_orig[1] = y
            # print(f"Rotating: azim={azim}, elev={elev}")
            glutPostRedisplay()

        elif self.zoom:
            self.distance += 0.01 * (y - self.mouse_orig[1])
            self.mouse_orig[1] = y
            # print(f"Zooming: distance={distance}")
            glutPostRedisplay()

        elif self.translation:
            v = [np.sin(self.azim * np.pi / 180), np.cos(self.azim * np.pi / 180)]
            vp = [v[1], -v[0]]
            dx = (x - self.mouse_orig[0])
            dy = (y - self.mouse_orig[1])
            self.origine[0] += 0.1 * (dx * vp[0] - dy * v[0])
            self.origine[1] += 0.1 * (dx * vp[1] - dy * v[1])
            self.mouse_orig[0] = x
            self.mouse_orig[1] = y
            # print(f"Translating: origine={origine}")
            glutPostRedisplay()

    def mouse(self,button, state, x, y):
        """
        mouse management
        """

        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                if glutGetModifiers() == GLUT_ACTIVE_CTRL:
                    print("Left button with CTRL pressed: Zoom mode")
                    self.mouse_orig[1] = y
                    self.zoom = True
                    self.rotate = False
                    self.translation = False
                else:
                    print("Left button pressed: Rotate mode")
                    self.mouse_orig[0] = x
                    self.mouse_orig[1] = y
                    self.rotate = True
                    self.zoom = False
                    self.translation = False
            else:
                print("Left button released")
                self.rotate = False

        elif button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN:
                if glutGetModifiers() == GLUT_ACTIVE_CTRL:
                    print("Right button with CTRL pressed: Translation mode")
                    self.mouse_orig[0] = x
                    self.mouse_orig[1] = y
                    self.translation = True
                    self.rotate = False
                    self.zoom = False
                else:
                    print("Right button pressed: Zoom mode")
                    self.mouse_orig[1] = y
                    self.zoom = True
                    self.rotate = False
                    self.translation = False
            else:
                print("Right button released")
                self.zoom = False
                self.translation = False

        self.is_paused = False

    def toggle_pause(self,):
        self.is_paused = not self.is_paused

    def key(self,key, x, y):
        if key == b'p':  # Assuming pressing 'p' should toggle pause
            self.toggle_pause()