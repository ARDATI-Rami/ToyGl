#!/usr/bin/env python
# -*- coding:Utf-8 -*-
# pyToyGL
# J. Averseng
# LMGC - Univ. Montpellier - France
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import time, random, math
import uuid
import hashlib
import matplotlib.pyplot as plt
import logging
from itertools import permutations,product
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import KDTree,ConvexHull
import dill
from  natsort import humansorted
# Global variable
window = 0
screenSize = [640, 480]
previousTime = time.time()
frameCounter = 0
frameCounterTimer = time.time()
delta_time_ccc = 1
distance = 20
elev = 0
azim = 0
rotate = False
zoom = False
translation = False
origine = [0., 0., 0.]
mouse_orig = [0., 0., 0.]

t = 0
dt = 0.002 ** 1
t_div_elem = 0

flags = {"dynamic_relaxation": False,
         "cell-cell_contact": True, 'node-node_contact': False}
params = {"gravity": -200*0, "medium_viscosity": 0}

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def midpoint(v1, v2):
    return (v1 + v2) / 2

def create_icosphere(depth=2):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    vertices = np.array([
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ], dtype=np.float64)

    vertices = np.array([normalize(v) for v in vertices])

    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ])

    def subdivide(vertices, faces, depth):
        vertices = list(vertices)
        for _ in range(depth):
            new_faces = []
            midpoint_cache = {}

            def cached_midpoint(v1, v2):
                key = tuple(sorted((v1, v2)))
                if key not in midpoint_cache:
                    midpoint_cache[key] = len(vertices)
                    mid = normalize(midpoint(np.array(vertices[v1]), np.array(vertices[v2])))
                    vertices.append(mid)
                return midpoint_cache[key]

            for tri in faces:
                v1, v2, v3 = tri
                a = cached_midpoint(v1, v2)
                b = cached_midpoint(v2, v3)
                c = cached_midpoint(v3, v1)
                new_faces.extend([
                    [v1, a, c],
                    [v2, b, a],
                    [v3, c, b],
                    [a, b, c],
                ])
            faces = new_faces

        return np.array(vertices), np.array(faces)

    vertices, faces = subdivide(vertices, faces, depth)

    return np.array(vertices), np.array(faces)



# Useful functions
def squared_norm(vec):
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]

def norm(vec):
    return np.sqrt(squared_norm(vec))


# Define ANSI escape sequences for colors
COLORS = {
    'reset': '\033[0m',      # Reset to default color
    'black': '\033[30m',     # Black text
    'red': '\033[31m',       # Red text
    'green': '\033[32m',     # Green text
    'yellow': '\033[33m',    # Yellow text
    'blue': '\033[34m',      # Blue text
    'magenta': '\033[35m',   # Magenta text
    'cyan': '\033[36m',      # Cyan text
    'white': '\033[37m',     # White text
    'bright_black': '\033[90m',  # Bright Black (also known as Grey)
    'bright_red': '\033[91m',    # Bright Red
    'bright_green': '\033[92m',  # Bright Green
    'bright_yellow': '\033[93m', # Bright Yellow
    'bright_blue': '\033[94m',   # Bright Blue
    'bright_magenta': '\033[95m',# Bright Magenta
    'bright_cyan': '\033[96m',   # Bright Cyan
    'bright_white': '\033[97m',  # Bright White
    'grey': '\033[90m',          # Grey (alias for Bright Black)
}

# Define a new log level
SUPERIOR_INFO = 25
logging.addLevelName(SUPERIOR_INFO, 'SUPERIOR_INFO')
log_file_path = 'debug.log'

# Clear the contents of the debug log file
with open(log_file_path, 'w'):
    pass  # Opening in 'w' mode truncates the file, no need to write anything
class DebugOnlyFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG

def superior_info(self, message, *args, **kws):
    # Wrap the message with ### for SUPERIOR_INFO
    message = f"### {message} ###"
    self._log(SUPERIOR_INFO, message, args, **kws) if self.isEnabledFor(SUPERIOR_INFO) else None

# Extend the Logger class to include the superior_info method
logging.Logger.superior_info = superior_info

def dont_debug(self, message, *args, **kws):
    # This method intentionally does nothing
    pass

# Extend the Logger class to include the dont_debug method
logging.Logger.dont_debug = dont_debug
# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all logs

# Filter for file handler to allow only DEBUG messages

# Create a file handler for debug logs
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.addFilter(DebugOnlyFilter())  # Apply the DebugOnlyFilter

# Define a file formatter and set it to the file handler
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Add file handler to the logger
logger.addHandler(file_handler)

# Create a console handler with a custom formatter
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # This will handle INFO and SUPERIOR_INFO

# Custom formatter with colors and ### for SUPERIOR_INFO
class CustomFormatter(logging.Formatter):
    def format(self, record):
        level = record.levelno
        if level == logging.INFO:
            color = COLORS['bright_white']
        elif level == SUPERIOR_INFO:
            color = COLORS['blue']
            record.msg = f"### {record.msg} ###"  # Add ### directly here for SUPERIOR_INFO messages
        else:
            color = COLORS['reset']
        message = super().format(record)
        return f"{color}{message}{COLORS['reset']}"

# Set the custom formatter to the console handler
console_formatter = CustomFormatter('%(message)s')
console_handler.setFormatter(console_formatter)
# Add console handler to the logger
logger.addHandler(console_handler)

# Example usage
logger.info('This is a regular info message.')  # Will be displayed in grey
logger.superior_info('Example of Superior Info')  # Will be displayed in blue with ### prefix and suffix
logger.debug('This debug message will go to the file.')
logger.dont_debug('This message will not be logged anywhere.')



class Node:
    """
    A class to represent a node in a simulation.

    Attributes:
        position (list or numpy.array): The position of the node in space.
        velocity (list or numpy.array): The velocity of the node.
        properties (dict): A dictionary to hold various properties of the node, such as mass, radius, etc.
    """
    _next_id = 0  # Class-level attribute to keep track of the next available sequential ID

    @classmethod
    def default_properties(cls):
        """
        Returns the default properties for a Node instance.

        Returns:
            dict: A dictionary of default properties.
        """
        return {'radius': 0.1, 'mass': 0.2, 'stiffness': 1000.0, 'color': (1.0, 0.0, 0.0)}

    def __init__(self, position, velocity=None, properties=None):
        """
        Initialize a Node instance.

        Args:
            position (list or numpy.array): The initial position of the node.
            velocity (list or numpy.array, optional): The initial velocity of the node. Defaults to [0, 0, 0].
            properties (dict, optional): Additional properties of the node (e.g., mass, radius). Defaults to an empty dict.
        """
        self.position = np.array(position)
        self.velocity = np.array(velocity) if velocity is not None else np.zeros(3)
        self.properties = properties if properties is not None else Node.default_properties()
        # Assign a sequential unique ID to this instance
        self.sequential_id = Node._next_id
        Node._next_id += 1

        # Assign a globally unique ID to this instance
        self.uuid = uuid.uuid4()

        self.forces = np.zeros(3)  # Initialize the forces as a zero vector
        self.pressure_forces = np.zeros(3)  # Initialize the pressure forces as a zero vector
        self.is_blocked = False  # attribute to indicate if the node is blocked


    def __repr__(self):
        return (f"Node {self.sequential_id} "
                f"Position: {self.position}, Velocity: {self.velocity}")

    def update_position(self, delta_t):
        """
        Update the node's position based on its velocity and a time step.

        Args:
            delta_t (float): The time step for the update.
        """
        # Assuming 'position' and 'velocity' are list-like or numpy arrays
        self.position = [self.position[i] + self.velocity[i] * delta_t for i in range(len(self.position))]

    def set_property(self, key, value):
        """
        Set or update a property of the node.

        Args:
            key (str): The property key.
            value: The value of the property.
        """
        self.properties[key] = value

    def get_property(self, key):
        """
        Get a property of the node.

        Args:
            key (str): The property key.

        Returns:
            The value of the property, or None if the property is not set.
        """
        return self.properties.get(key)

    def reset_forces(self):
        """Reset the forces to zero."""
        self.forces = np.zeros(3)
        self.pressure_forces = np.zeros(3)

    def add_force(self, additional_force):
        """
        Add a force vector to the node's existing forces.

        Args:
            additional_force (numpy.array): The force vector to be added.
        """
        self.forces += additional_force
    def ground_contact_force(self):
        """
        Calculate the ground contact force if the node is below a certain level.

        The method checks if the node's z-coordinate, minus its radius, is below zero,
        indicating penetration into the ground. A ground contact force is then applied,
        calculated using the node's stiffness property and the depth of penetration.


        Returns:
            numpy.array: The ground contact force.
        """
        dz = self.position[2] - self.properties['radius']
        if dz < 0.:
            return np.array([0., 0., self.properties['stiffness'] * -dz])
        else:
            return np.array([0., 0., 0.])

    def gravity_force(self, gravity):
        """
        Calculate the gravity force acting on the node.

        This method takes the gravitational acceleration as an argument and calculates
        the gravitational force by multiplying it by the node's mass.

        Args:
            gravity (float): The gravitational acceleration.

        Returns:
            numpy.array: The gravity force.
        """
        return np.array([0., 0., gravity * self.properties['mass']])

    def viscous_damping_force(self, medium_viscosity):
        """
        Calculate the viscous damping force based on the node's velocity.

        The damping force is proportional to the node's velocity and the viscosity
        of the medium. The negative sign indicates that the damping force opposes
        the direction of the velocity.

        Args:
            medium_viscosity (float): The viscosity of the medium.

        Returns:
            numpy.array: The viscous damping force.
        """
        return -1*self.velocity * medium_viscosity

    def block(self):
        """Block the node from moving."""
        self.is_blocked = True

    def unblock(self):
        """Unblock the node and allow it to move."""
        self.is_blocked = False

    def draw(self, selected=False, in_contact=False):
        """
        Draws the node as a sphere using OpenGL commands.

        Args:
            selected (bool, optional): Indicates if the node is selected. Defaults to False.
            in_contact (bool, optional): Indicates if the node is in contact. Defaults to False.
        """
        glColor3f(*self.properties['color'])  # Set the color of the node
        radius = self.properties['radius']  # Get the radius of the node
        # Increase the size if the node is selected or in contact
        if selected:
            glColor3f(1, 1, 0)  # Yellow color for selected nodes
            radius *= 1.1
        if in_contact:
            glColor3f(1, 0, 0)  # Red color for nodes in contact
            radius *= 1.5

        glColor3f(1.0, 0.0, 1.0)
        # Draw the sphere at the node's position
        draw_sphere(self.position, radius)


    def is_blocked(self):
        """Check if the node is blocked."""
        return self.is_blocked

class Filament:
    """
    A class to describe a filament, which acts as an edge in a cell.
    """
    _next_id = 0  # Class-level attribute to keep track of the next available ID
    _existing_filaments = set()  # Class-level set to track unique filaments

    @classmethod
    def default_properties(cls):
        """
        Returns the default properties for a Node instance.

        Returns:
            dict: A dictionary of default properties.
        """
        return {'radius': 0.05, 'linear_mass': 0.2, 'rig_EA': 800, 'color': (0.0, 1.0, 0.0)}

    def __new__(cls, node1, node2, cell_id, properties=None,type='Structure'):
        """
        Create a new instance of Filament if it does not already exist.

        Args:
            node1 (Node): The first node of the filament.
            node2 (Node): The second node of the filament.
            cell_id (int): The cell id to which this filament belongs.
            properties (dict, optional): The properties of the filament.

        Returns:
            Filament or None: A new Filament instance or None if it already exists.
        """
        # Create a unique identifier based on sorted node UUIDs
        node_ids = sorted([str(node1.sequential_id), str(node2.sequential_id)])
        combined_ids = ''.join(node_ids)

        # Check if this filament already exists
        if combined_ids in cls._existing_filaments:
            logger.info(f"Filament with nodes {node1.sequential_id} and {node2.sequential_id} already exists. Initialization aborted.")
            return None

        # Add the filament to the set of existing filaments
        cls._existing_filaments.add(combined_ids)

        return super(Filament, cls).__new__(cls)
    def __init__(self, node1, node2, cell_id, properties=None, type ="Cell"):
        """
        Initialize a filament with two Node instances and a cell ID.

        Args:
            node1 (Node): The first node of the filament.
            node2 (Node): The second node of the filament.
            cell_id (int): The cell id to which this filament belongs.
        """
        logger.info(f"Initialising new filament number : {Filament._next_id+1}, ({node1.sequential_id}, {node2.sequential_id}) ...")
        self.node1 = node1
        self.node2 = node2
        self.nodes = [self.node1,self.node2]
        self.cell_id = cell_id

        # Create unique IDs for the filament based on node IDs
        # Option 1: Concatenating sequential IDs of the nodes
        self.unique_id_concat = f"{self.node1.sequential_id}-{self.node2.sequential_id}"
        # Option 2: Hashing a tuple of sequential IDs
        combined_uuids = str(node1.uuid) + str(node2.uuid)
        self.uuid = uuid.UUID(hashlib.md5(combined_uuids.encode()).hexdigest())
        self.cell_id = cell_id
        self.node_properties_list = [{} for _ in range(4)]
        # self.filament_properties_list = [{} for _ in range(4)]
        self.filament_properties_list = {'radius': 0.2, 'mass': 0.2, 'stiffness': 1000.0, 'color': (0.2, 0.2, 0.2)}


        self.properties = properties if properties is not None else Filament.default_properties()
        self.node_families=[]
        self.lfree_new = 2
        # Assign a unique ID to this instance and increment the class-level ID counter
        self.sequential_id = Filament._next_id
        Filament._next_id += 1

        self.ldivide = 1 # threshold length for filament division
        self.age_steps = 0  # Initialize age in simulation steps


    def __repr__(self):
        return (f"Filament {self.sequential_id}:({self.unique_id_concat}): "
                f"(NodeA ID:{self.node1.sequential_id}, Position: {self.node1.position}, "
                f"NodeB ID:{self.node2.sequential_id} Position: {self.node2.position}, "
                f"Cell ID: {self.cell_id})")

    @property
    def node_positions(self):
        """
        Return the positions of node1 and node2 of the filament.

        Returns:
            tuple: A tuple containing the positions of node1 and node2.
        """
        return (self.node1.position, self.node2.position)
    @property
    def length(self):
        """Calculate the current length of the filament."""
        return norm(np.array(self.node2.position) - np.array(self.node1.position))

    @property
    def vector(self):
        """Calculate the current vector of the filament."""
        return np.array(self.node2.position) - np.array(self.node1.position)

    def increment_age(self):
        """Increment the age of the filament by one simulation step."""
        # logger.info(f"Incrementing age for filament : {self.sequential_id}, new age : {self.age_steps} + 1")
        self.age_steps += 1
    def redefine_lfree(self):
        """ Redefine l free as current filament length"""
        self.lfree_new = self.length
    def was_created_recently(self):
        """Determine if the filament was created 'recently'."""
        return self.age_steps < 1  # Adjust threshold as needed

    def midpoint(self):
        """
        Calculate and return the midpoint of the filament.

        Returns:
            numpy.array: The midpoint of the filament.
        """
        return (self.node1.position + self.node2.position) / 2

    def replace_node2(self, new_node):
        """
        Replace the second node of the filament with a new node.

        Args:
            new_node (Node): The new node to replace the current second node.
        """
        self.node2 = new_node
        # Update the unique IDs if needed
        self.unique_id_concat = f"{self.node1.sequential_id}-{self.node2.sequential_id}"
        combined_uuids = str(self.node1.uuid) + str(self.node2.uuid)
        self.uuid = uuid.UUID(hashlib.md5(combined_uuids.encode()).hexdigest())

    def made_of_this(self, node):
        """
        Check if a given node belongs to this filament.

        Args:
            node (Node): The node to check.

        Returns:
            bool: True if the node is part of the filament, False otherwise.
        """
        return node == self.node1 or node == self.node2

    def flip_in_place(self):
        """
        Flip the filament by reversing the order of its nodes, modifying the object in place.
        """
        # Swap the nodes

        self.node1, self.node2 = self.node2, self.node1 #TODO  # do we still need you

        self.unique_id_concat = f"{self.node1.sequential_id}-{self.node2.sequential_id}"
        logger.info(f"After flipping Node 1 is :  {self.node1} and Node 2 is : {self.node2}")
    def get_velocity(self):
        """
        Return the velocity vector of the filament

        Returns:
            (np.array): Filament Velocity array
        """

        return 0.5*(self.node1.velocity+self.node2.velocity)


    def elastic_force(self):
        """
        Calculate the elastic force of the filament.

        Returns:
            numpy.array: The elastic force vector of the filament.
        """
        l0 = self.lfree_new  # Free length of the filament
        l = self.length  # Current length of the filament
        v_filament = self.vector  # Vector representing the filament
        EA = self.properties["rig_EA"]  # Stiffness of the filament

        # Calculate the elastic force
        if l != 0:  # To avoid division by zero
            v_force = v_filament * EA * (l - l0) / (l0 * l)
        else:
            v_force = np.zeros_like(v_filament)

        return v_force


    def draw_cylinder(self):
        """
        Draws the filament as a cylinder using OpenGL commands. This function calculates the
        direction vector of the filament and determines the appropriate rotation to align a
        cylinder along this vector. It handles special cases where the filament is aligned
        with or against the reference direction (typically the Z-axis).

        The filament's properties such as radius and the positions of its nodes are used to
        determine the size and placement of the cylinder.
        """
        # Direction vector (normalized)
        direction = self.vector / self.length

        # Reference vector (Z-axis)
        reference = np.array([0.0, 0.0, 1.0])

        # Rotation axis (cross product)
        axis = np.cross(reference, direction)

        # Rotation angle (arccosine of dot product)
        angle = math.acos(np.dot(reference, direction))

        # Convert angle to degrees
        angle_degrees = math.degrees(angle)

        # Starting point of the cylinder
        pt1 = self.node1.position

        # Length of the cylinder
        length = self.length

        # Cylinder radius
        radius = self.properties['radius']

        # Set up the transformation and rotation
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(pt1[0], pt1[1], pt1[2])

        if np.linalg.norm(axis) > 1e-6:  # Check if the axis is non-zero
            glRotatef(angle_degrees, axis[0], axis[1], axis[2])
        else:
            # Special case: direction is parallel to reference, no rotation needed
            if np.dot(direction, reference) < 0:
                # 180-degree rotation around any axis perpendicular to reference
                glRotatef(180, 1, 0, 0)

        # Create and draw the cylinder
        quad = gluNewQuadric()
        gluQuadricOrientation(quad, GLU_OUTSIDE)
        gluQuadricDrawStyle(quad, GLU_FILL)
        gluCylinder(quad, radius, radius, length, 8, 1)

        # Restore the matrix state
        glPopMatrix()

    def draw(self, selected=False):
        """
        Draws the filament as a cylinder using OpenGL commands.

        Args:
            selected (bool, optional): Indicates if the filament is selected. Defaults to False.
        """
        glColor3f(*self.properties['color'])  # Set the color of the filament
        radius = self.properties['radius']  # Get the radius of the filament

        # Increase the size if the filament is selected
        if selected:
            glColor3f(1.0,0.0 , 0.0)  # Yellow color for selected filaments
            self.properties['radius'] = 0.025

        # Draw the cylinder representing the filament
        self.draw_cylinder()

        # Reset color to default (optional)
        glColor3f(1, 1, 1)

class Facet:
    """
    A class to describe a triangular facet in a cell, formed by three filaments.
    """
    _next_id = 0  # Class-level attribute for sequential IDs

    def __init__(self, filament1, filament2, filament3):
        """
        Initialize a Facet object with three filaments.

        Parameters:
        filament1: The first filament of the triangular facet.
        filament2: The second filament of the triangular facet.
        filament3: The third filament of the triangular facet.
        """
        # logger.info(f"Initialising a facet object")
        self.filament1 = filament1
        self.filament2 = filament2
        self.filament3 = filament3
        self.filaments = [self.filament1, self.filament2, self.filament3]

        # Assign a sequential unique ID to this instance
        self.sequential_id = Facet._next_id

        Facet._next_id += 1

        # Combine the UUIDs of the filaments
        combined_uuids = ''.join(str(filament.uuid) for filament in self.filaments)

        # Create a UUID for the facet based on the combined UUIDs of the filaments
        self.uuid = uuid.UUID(hashlib.md5(combined_uuids.encode()).hexdigest())
        self.tetrahedron_center = None
        self.is_normal_outward()



    def __repr__(self):
        """
        Return a string representation of the Facet object using the existing __repr__ method of Filaments.
        """
        facet_info = f"Facet {self.sequential_id} details:\n"
        filaments_info = "\n".join([f"{idx+1}: {filament}" for idx, filament in enumerate(self.filaments)])
        return f"{facet_info}{filaments_info}"


    def get_unique_nodes_ordre(self,verbose=False):
        """
        Retrieves the three unique nodes that form the facet, ensuring they are ordered
        based on the connectivity of the filaments.
        Frist append node 1 of fil1, than the shared node of fil 2 and fil3, than append the node2 of fil 1

        Returns:
            list: A list containing three unique Node instances forming the facet.
        """
        # first add the fil1.node1 to nodes
        nodes = [self.filament1.node1]
        if verbose:
            logger.info(f"get_unique_nodes")
            for i,fil in enumerate(self.filaments):
                logger.info(f'Fialemnts {i+1}, node 1 :{fil.node1.sequential_id} and node 2 : {fil.node2.sequential_id}!')
        if self.filament2.node1.sequential_id == self.filament3.node1.sequential_id:
            if verbose:
                logger.info(f'Node 1 is shared between fil2 and fil3')
            # add the second node of filament 1
            nodes.append(self.filament2.node1)
            # add the second node of filament 1
            nodes.append(self.filament1.node2)

        if self.filament2.node2.sequential_id == self.filament3.node2.sequential_id:
            if verbose:
                logger.info(f'Node 2 is shared between fil2 and fil3')
            # add the second node of filament 1
            nodes.append(self.filament1.node2)
            # add the shared one now
            nodes.append(self.filament2.node2)
        if self.filament2.node2.sequential_id == self.filament3.node1.sequential_id:
            if verbose:
                logger.info(f'Cross Node is shared between fil2 and fil3')
            # add the second node of filament 1
            nodes.append(self.filament1.node2)
            # add the shared one now
            nodes.append(self.filament2.node2)
        # Ensure only three unique nodes are considered
        unique_nodes = list(set(nodes))  # Convert to set and back to list to remove duplicates
        if len(unique_nodes) != 3:
            logger.info(f"Error in facet : {self.sequential_id}")
            logger.info(f"Error in facet : {self.__repr__()}")
            logger.info(f" The error happened here, nodes : {nodes}!!!!")
            raise ValueError(f"Expected 3 unique nodes to form a facet, but got {len(unique_nodes)}.")
        # logger.info(f"unique nodes :{nodes}")

        return nodes

    def get_unique_nodes(self,verbose=False):
        """
        Retrieves the three unique nodes that form the facet.

        Returns:
            list: A list containing three unique Node instances forming the facet.
        """
        # Initialize an empty set to store unique nodes
        unique_nodes = set()

        # Add nodes from each filament to the set
        # Since a set automatically eliminates duplicates, each node will be added only once
        unique_nodes.add(self.filament1.node1)
        unique_nodes.add(self.filament1.node2)
        unique_nodes.add(self.filament2.node1)
        unique_nodes.add(self.filament2.node2)
        unique_nodes.add(self.filament3.node1)
        unique_nodes.add(self.filament3.node2)

        # Ensure only three unique nodes are considered
        if len(unique_nodes) != 3:
            logger.info(f"Error in facet : {self.sequential_id}")
            logger.info(f"Error in facet : {self.__repr__()}")
            logger.info(f" The error happened here, nodes : {unique_nodes}!!!!")
            raise ValueError(f"Expected 3 unique nodes to form a facet, but got {len(unique_nodes)}.")
        # logger.info(f"unique nodes :{nodes}")
        # Convert the set to a list to make it subscriptable
        unique_nodes_list = list(unique_nodes)
        return unique_nodes_list

    def is_normal_outward(self):
        if self.tetrahedron_center is not None:
            # Initially set a flag to False to enter the while loop
            normal_is_outward = False
            count = 0
            while not normal_is_outward and count < 6:
                # Assuming calculate_normal is a method that calculates the normal of a self
                normal = self.calculate_normal()

                # Choose a point on the self (e.g., the position of the first node of the first filament)
                point_on_self = self.filaments[0].node1.position
                # Calculate the vector from the center to the point on the self
                center_to_point_vector = point_on_self - self.tetrahedron_center

                # Calculate the dot product
                dot_product = np.dot(normal, center_to_point_vector)

                if dot_product > 0:
                    # logger.info(f"For facet : {self.sequential_id} the normal was outward : {True}")
                    normal_is_outward = True  # Update the flag to exit the loop
                else:
                    self.reverse_filaments()  # Reverse the filaments if not outward
                    count += 1
                    # If, after 6 attempts, the normal is still not outward, raise an error
            if not normal_is_outward:
                raise ValueError("After 6 attempts, the normal is still not outward.")
    def reverse_filaments(self):
        """
        Swap the first and second filaments without affecting the third.
        """
        # Directly swap filament1 and filament2
        self.filament1, self.filament2 = self.filament2, self.filament1

        # Update the filaments list to reflect the new order
        self.filaments = [self.filament1, self.filament2, self.filament3]

    def calculate_normal(self):
        """Calculate the normal vector of a facet defined by two filaments."""

        return np.cross(self.filament1.vector, self.filament2.vector)

    def get_facets_vectors(self):
        """
        Return the three filament-vectors of the facet.

        Returns:
            numpy.array: The three filament-vectors  of the facet.
        """

        # Get the vectors of the filaments
        v1 = self.filament1.vector
        v2 = self.filament2.vector
        v3 = self.filament3.vector

        return v1,v2,v3
    def area(self):
        """
        Calculate and return the area of the triangular facet.

        Returns:
            float: The area of the facet.
        """
        # Get the vectors of the filaments forming two sides of the triangle
        v1, v2, _ = self.get_facets_vectors()

        # Calculate the cross product of v1 and v2
        cross_product = np.cross(v1, v2)

        # The area of the triangle is half the magnitude of the cross product
        area = 0.5 * np.linalg.norm(cross_product)
        return area

    def contains_filament(self, filament):
        """
        Check if the facet contains the given filament.

        Args:
            filament (Filament): The filament to check.

        Returns:
            bool: True if the facet contains the filament, False otherwise.
        """
        return filament in self.filaments


    def detect_disconnected_filament(self, target_filament):
        """
        Detects if there are filaments within the facet that aren't connected to the specified target filament.

        Args:
            target_filament: The filament to check connectivity against.

        Returns:
            tuple: A tuple containing the disconnected filament object and its index in the filament list,
                   or (None, None) if all filaments are connected to the target filament.
        """
        target_nodes = {target_filament.node1, target_filament.node2}
        # for fil in self.filaments:
        #     logger.info(f"fil :{fil.sequential_id} and age : {fil.age_steps}")

        # Check each filament for connectivity to target_filament
        for i, filament in enumerate(self.filaments):
            if filament == target_filament:
                continue  # Skip the target filament itself

            # Check if filament shares any node with target_filament
            filament_nodes = {filament.node1, filament.node2}
            if target_nodes.isdisjoint(filament_nodes):
                logger.info(f"Filament  {i} :{filament} is not connected to the target filament. Age : {filament.age_steps}")
                removed_filament = self.filaments.pop(i)
                return removed_filament, i  # Return the disconnected filament and its index
        return None, None

    def add_filament(self, filament, index):
        """
        Adds a filament to the facet at the specified position using insert method.

        Args:
            filament (Filament): The filament to be added.
            index (int): The zero-based index at which the filament should be inserted.

        Returns:
            None
        """

        # Ensure the index is within bounds or at the end of the list
        if 0 <= index <= 3:
            self.filaments.insert(index, filament)
            logger.info(f"Filament added successfully at position ID: {index}")
            self.filament1 = self.filaments[0]
            self.filament2 = self.filaments[1]
            self.filament3 = self.filaments[2]
            self.filaments = [self.filament1, self.filament2, self.filament3]

        else:
            logger.info(f"Invalid index: {index}. Filament not added.")

    @staticmethod
    def draw_triangle(pt1, pt2, pt3, color):
        """
        Draw a triangle using the given vertex positions.

        Args:
            pt1 (np.array): Position of the first vertex.
            pt2 (np.array): Position of the second vertex.
            pt3 (np.array): Position of the third vertex.
            color (tuple): Color of the triangle (R, G, B, Alpha).
        """
        glDisable(GL_CULL_FACE)
        glBegin(GL_TRIANGLES)
        glColor4f(*color)  # Set the color for the triangle
        glVertex3f(*pt1)
        glVertex3f(*pt2)
        glVertex3f(*pt3)
        glEnd()

    def draw(self, color=(0, 0, 1, 0.5)):
        """
        Draws the facet as a triangle using OpenGL commands.

        Args:
            color (tuple, optional): The color of the facet (R, G, B, Alpha). Defaults to (0, 1, 0, 0.5).
        """
        # Get the positions of the unique nodes forming the triangle
        # logger.info(f"Drawing Facet : {self.sequential_id}")
        # logger.info(self.__repr__)
        node1, node2, node3 = self.get_unique_nodes()
        # Draw the triangle
        self.draw_triangle(node1.position, node2.position, node3.position, color)

class Cell:
    """
    A class to describe a 3D cell
    """
    liste_tri3 = []  # list of 3 nodes ids describing a facet, facing outside
    volume = 1.  # calculé à chaque pas de temps
    dp_dv = 10.  # rigidité
    _next_id = 0
    pressure = 0

    def __init__(self):
        """
        Init cell with a single tetraedra
         """

        self.initialize_basic_cell()
        combined_uuids = ''.join(str(facet.uuid) for facet in self.facets)
        self.uuid = uuid.UUID(hashlib.md5(combined_uuids.encode()).hexdigest())
        self.sequential_id = Cell._next_id
        Cell._next_id += 1

        self.division_counter = 0
        # self.is_normals_outward()
        self.volume0 = self.calculate_volume()
        logger.debug(f"Volume 0 is : {self.volume0}")

    def initialize_basic_cell(self, initial_shape='Icosahedron', depth=0):
        def find_filaments_for_facet(node_indices, filaments):
            node_set = list(node_indices)
            found_filaments = []
            for filament in filaments:
                set_fil = [filament.node1.sequential_id, filament.node2.sequential_id]
                if filament.node1.sequential_id in node_set and filament.node2.sequential_id in node_set:
                    found_filaments.append(filament)
            if len(found_filaments) != 3:
                print("Error found less than 3 filaments")
            return found_filaments

        if initial_shape == 'Tetra':
            pt0 = [0, 0, 2]
            l0 = 1
            # First create the nodes
            pt1 = pt0 + np.array([l0, 0, 0])
            pt2 = pt0 + np.array([0, l0, 0])
            pt3 = pt0 + np.array([0, 0, l0])

            Node0 = Node(position=pt0)
            Node1 = Node(position=pt1)
            Node2 = Node(position=pt2)
            Node3 = Node(position=pt3)

            # Create filaments
            fil_0 = Filament(node1=Node0, node2=Node1, cell_id=0)  # 01
            fil_1 = Filament(node1=Node0, node2=Node2, cell_id=0)  # 02
            fil_2 = Filament(node1=Node0, node2=Node3, cell_id=0)  # 03
            fil_3 = Filament(node1=Node1, node2=Node2, cell_id=0)  # 12
            fil_4 = Filament(node1=Node1, node2=Node3, cell_id=0)  # 13
            fil_5 = Filament(node1=Node2, node2=Node3, cell_id=0)  # 32
            # Create facets
            facet0 = Facet(fil_1, fil_0, fil_3)
            facet1 = Facet(fil_0, fil_2, fil_4)
            facet2 = Facet(fil_5, fil_2, fil_1)
            facet3 = Facet(fil_3, fil_4, fil_5)
            self.facets = [facet0, facet1, facet2, facet3]

        elif initial_shape == 'Icosahedron':
            print(f'Icosahedron intialisation')
            vertices, faces = create_icosphere(depth)

            # Create nodes
            nodes = [Node(position=vertex) for vertex in vertices]
            print(f"nodes : {len(nodes)}")

            facets_indices = faces
            print(f"facets_indices : {facets_indices}")

            cell_id = self._next_id
            # Create the facets using the find_filaments_for_facet function
            facets = []
            # create all the unique filaments
            all_filaments = set()
            for facet_indices in facets_indices:
                n1, n2, n3 = facet_indices
                print(f"n123 : {n1}, {n2} , {n3}")
                filament_for_facet = []
                # create the filament for every combination of facet indices
                fil_0 = Filament(node1=nodes[n1], node2=nodes[n2], cell_id=cell_id)
                fil_1 = Filament(node1=nodes[n2], node2=nodes[n3], cell_id=cell_id)
                fil_2 = Filament(node1=nodes[n3], node2=nodes[n1], cell_id=cell_id)
                if fil_0 is not None:
                    filament_for_facet.append(fil_0)
                if fil_1 is not None:
                    filament_for_facet.append(fil_1)
                if fil_2 is not None:
                    filament_for_facet.append(fil_2)
                all_filaments.update(filament_for_facet)
                print(f"all_filaments : {len(all_filaments)}")
                for fil in all_filaments:
                    fil.redefine_lfree()
            print(f"Number of initialized filaments: {len(all_filaments)}")
            if cell_id > 0:
                # update facet indices
                facets_indices = [(x + 12 * cell_id, y + 12 * cell_id, z + 12 * cell_id) for x, y, z in facets_indices]
            for indices in facets_indices:
                filaments_for_facet = find_filaments_for_facet(indices, all_filaments)
                print(f"filament for facet : {filament_for_facet}")
                facet = Facet(*filaments_for_facet)
                facets.append(facet)

            self.facets = facets
            print(f"Number of facets: {len(self.facets)}")
    # def initialize_basic_cell(self,intial_shape='Icosahedron'):
    #     """
    #     Initializes a basic cell structure with predefined nodes, filaments, and facets.
    #     It creates a tetrahedral cell configuration as a starting point for the simulation.
    #     The cell is defined by 4 nodes positioned in a way that forms a tetrahedron.
    #     This method also visualizes the cell structure and its normals to ensure correctness.
    #     """
    #
    #     def find_filaments_for_facet(node_indices, filaments):
    #         node_set = list(node_indices)
    #         print(f"node set : {node_set}")
    #         found_filaments = []
    #         for filament in filaments:
    #             set_fil = [filament.node1.sequential_id, filament.node2.sequential_id]
    #             if filament.node1.sequential_id in node_set and filament.node2.sequential_id in node_set:
    #                 print(f"found fil : {filament.sequential_id}, {set_fil} for node indices {node_indices}")
    #                 found_filaments.append(filament)
    #         if found_filaments != 3:
    #             print("Error found less than 3 fil")
    #         print(f"found fil:{found_filaments}")
    #         return found_filaments
    #
    #     if intial_shape == 'Tetra':
    #         pt0 = [0, 0, 2]
    #         l0 = 1
    #         # First create the nodes
    #         pt1 = pt0 + np.array([l0, 0, 0])
    #         pt2 = pt0 + np.array([0, l0, 0])
    #         pt3 = pt0 + np.array([0, 0, l0])
    #
    #         Node0 = Node(position=pt0)
    #         Node1 = Node(position=pt1)
    #         Node2 = Node(position=pt2)
    #         Node3 = Node(position=pt3)
    #
    #         logger.info("Initialising the Node:")
    #         for nd in [Node0, Node1, Node2, Node3]:
    #             logger.info(f"{nd}")
    #         # Create filaments
    #         fil_0 = Filament(node1=Node0, node2=Node1, cell_id=0) #01
    #         fil_1 = Filament(node1=Node0, node2=Node2, cell_id=0) #02
    #         fil_2 = Filament(node1=Node0, node2=Node3, cell_id=0) #03
    #         fil_3 = Filament(node1=Node1, node2=Node2, cell_id=0) #12
    #         fil_4 = Filament(node1=Node1, node2=Node3, cell_id=0) #13
    #         fil_5 = Filament(node1=Node2, node2=Node3, cell_id=0) #32
    #         # Create facets
    #         facet0 = Facet(fil_1, fil_0, fil_3)
    #         facet1 = Facet(fil_0, fil_2, fil_4)
    #         facet2 = Facet(fil_5, fil_2, fil_1)
    #         facet3 = Facet(fil_3, fil_4, fil_5)
    #         self.facets = [facet0, facet1, facet2, facet3]
    #
    #     elif intial_shape == 'Icosahedron':
    #         phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    #         radius = 1  # You can adjust the radius as needed
    #         # Define the 12 vertices of the icosahedron
    #         vertices = [
    #             [-1, phi, 0],
    #             [1, phi, 0],
    #             [-1, -phi, 0],
    #             [1, -phi, 0],
    #             [0, -1, phi],
    #             [0, 1, phi],
    #             [0, -1, -phi],
    #             [0, 1, -phi],
    #             [phi, 0, -1],
    #             [phi, 0, 1],
    #             [-phi, 0, -1],
    #             [-phi, 0, 1],
    #         ]
    #         vertices = np.array(vertices) * radius / np.linalg.norm([1, phi, 0])  # Normalize to the desired radius
    #
    #         # Create nodes
    #         nodes = [Node(position=vertex) for vertex in vertices]
    #         print(f'Number of nodes : {len(nodes)}')
    #         facets_indices = [
    #             (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    #             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    #             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    #             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)
    #         ]
    #         cell_id = self._next_id
    #         # Create the facets using the find_filaments_for_facet function
    #         facets = []
    #         # create all the unique filaments 30 in total
    #         all_filaments = set()
    #         for facet_indices in facets_indices:
    #             n1,n2,n3 = facet_indices
    #             filament_for_facet = []
    #             # let's create the filament for every combination of facet indices
    #             fil_0 = Filament(node1=nodes[n1], node2=nodes[n2], cell_id=cell_id)
    #             fil_1 = Filament(node1=nodes[n2], node2=nodes[n3], cell_id=cell_id)
    #             fil_2 = Filament(node1=nodes[n3], node2=nodes[n1], cell_id=cell_id)
    #             if fil_0 is not None:
    #                 filament_for_facet.append(fil_0)
    #             if fil_1 is not None:
    #                 filament_for_facet.append(fil_1)
    #             if fil_2 is not None:
    #                 filament_for_facet.append(fil_2)
    #             all_filaments.update(filament_for_facet)
    #             for fil in all_filaments:
    #                 fil.redefine_lfree()
    #         print(f"Number of initialized filaments : {len(all_filaments)}")
    #         if cell_id > 0:
    #             # update facet indices
    #             facets_indices = [(x + 12*cell_id, y + 12*cell_id, z + 12*cell_id) for x, y, z in facets_indices]
    #         for indices in facets_indices:
    #             filaments_for_facet = find_filaments_for_facet(indices, all_filaments)
    #             facet = Facet(*filaments_for_facet)
    #             facets.append(facet)
    #
    #         self.facets = facets
    #         print(f"Number of facets : {len(self.facets)}")




    def translate_cell(self, translation_vector):
        """
        Translates the entire cell by a given vector.

        Args:
            translation_vector (list or np.array): A vector specifying the translation amount in each dimension.
        """
        for node in self.get_nodes():
            node.position += np.array(translation_vector)
            logger.info(f"Node {node.sequential_id} translated to new position: {node.position}")

    def get_nodes(self):
        """
        Returns a list of all unique nodes in the cell.

        Returns:
            list: A list of unique Node instances.
        """
        unique_nodes = set()
        unique_node_instances = []

        for filament in self.get_filaments():
            if filament.node1.uuid not in unique_nodes:
                unique_nodes.add(filament.node1.uuid)
                unique_node_instances.append(filament.node1)
            if filament.node2.uuid not in unique_nodes:
                unique_nodes.add(filament.node2.uuid)
                unique_node_instances.append(filament.node2)

        return unique_node_instances

    def get_filaments(self):
        """
        Returns a list of all unique filaments in the cell.

        Returns:
            list: A list of unique Filament instances.
        """
        unique_filaments = set()
        unique_filament_instances = []

        for facet in self.facets:
            for filament in facet.filaments:
                # Check if the filament has already been added
                if filament.uuid not in unique_filaments:
                    unique_filaments.add(filament.uuid)
                    unique_filament_instances.append(filament)

        return unique_filament_instances

    def center(self):
        """
        Calculate and return the centroid of the cell.

        Returns:
            numpy.array: The centroid of the cell.
        """
        nodes = self.get_nodes()
        if not nodes:
            return np.array([0, 0, 0])  # Default center if no nodes

        positions = np.array([node.position for node in nodes])
        centroid = np.mean(positions, axis=0)
        return centroid

    def is_normals_outward(self):
        tetrahedron_center = self.center()
        for facet in self.facets:
            # add the center of the tetrahedron to the facet
            facet.tetrahedron_center = tetrahedron_center
            # Assuming calculate_normal is a method that calculates the normal of a facet
            normal = facet.calculate_normal()

            # Choose a point on the facet (e.g., the position of the first node of the first filament)
            point_on_facet = facet.filaments[0].node1.position

            # Calculate the vector from the center to the point on the facet
            center_to_point_vector = point_on_facet - tetrahedron_center

            # Calculate the dot product
            dot_product = np.dot(normal, center_to_point_vector)
            if dot_product > 0:
                logger.info(f"For Facet : {facet.sequential_id} the normal was not outward, Reversing order ...")
                facet.reverse_filaments()



    # def calculate_volume(self):
    #     """
    #     Calculates the volume of the cell based on its facets.
    #     """
    #     vol = 0
    #     # Choose a reference point (e.g., the position of the first node in the first facet)
    #     ref_point = self.facets[0].get_unique_nodes()[0].position
    #     ref_point_id = self.facets[0].get_unique_nodes()[0].sequential_id
    #     # logger.debug(f"ref_point_id : {ref_point_id}")
    #     # logger.info(f"ref_point : {ref_point}")
    #
    #     for facet in self.facets:
    #         # Calculate the normal of the facet
    #         vn = facet.calculate_normal()
    #
    #         # Calculate the vector from the reference point to a point in the plane
    #         v01 = facet.filament1.node1.position - ref_point
    #
    #         # Calculate the area of the facet (triangle)
    #         base_area = 0.5 * norm(vn)
    #
    #         # Calculate the height from the reference point to the plane
    #         height = np.dot(vn, v01) / norm(vn)
    #
    #         vol += base_area * height / 3
    #
    #     self.volume = vol
    #     logger.debug(f"The difference in percentage : {(self.volume-self.calculate_volume2())/self.volume *100}")

    def redefine_volume0(self):
        """ Redefine volume 0 as the current cell volume"""
        self.volume0 = self.calculate_volume()
    def calculate_volume(self):
        """
        Calculate the volume of a 3D object defined by a list of 3D points.

        Parameters:
        points (np.ndarray): An array of shape (N, 3) where N is the number of points,
                             and each row represents the (x, y, z) coordinates of a point.

        Returns:
        float: The volume of the convex hull of the 3D object.
        """
        print(f"Calculating volume ")
        # Ensure points are in a numpy array of shape (N, 3)
        points = np.array([node.position for node in self.get_nodes()])
        # print(f"Points : {points}")
        # Ensure there are enough points and they are in 3D
        if points.shape[0] < 4 or points.shape[1] != 3:
            raise ValueError("There must be at least 4 points in 3D space.")

        try:
            hull = ConvexHull(points=points)
            self.volume = hull.volume
            # return hull.volume
        except Exception as e:
            print(f"Error calculating volume: {e}")
            return None
        return self.volume


    def calculate_pressure(self):
        """
        Calculates the pressure inside the cell based on its current volume.
        """
        # first Calculate Volume
        self.calculate_volume()
        F = len(self.get_filaments())
        # Calculate the pressure
        # self.pressure = 200
        self.pressure = (self.volume0 - self.volume)*self.dp_dv

    def apply_pressure_to_nodes(self):
        """
        Applies pressure forces to the nodes of the cell based on its facets.
        """

        for facet in self.facets:
            # Calculate the normal vector of the facet
            vn = facet.calculate_normal()
            # logger.info(f"vn : {vn}")
            # Calculate the force due to pressure on the facet
            force = self.pressure * vn * 0.5
            # logger.info(f"v_force : {force} and 1/3 v_force : {force / 3}")

            # Distribute the force to each node of the facet
            unique_nodes = facet.get_unique_nodes()
            for node in unique_nodes:
                node.pressure_forces += force / 3  # Assuming equal distribution to each node

    def plot_facets_with_normals(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Get all nodes and their positions
        nodes = self.get_nodes()
        positions = np.array([node.position for node in nodes])

        # Find min and max for each axis to set the plot limits
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()

        # Define a list of colors to cycle through. Add more colors if needed.
        colors = ['lightblue', 'red', 'yellow', 'green', 'purple', 'orange', 'cyan', 'magenta']
        color_index = 0

        for i, facet in enumerate(self.facets):
            # Get node positions
            pt0, pt1, pt2 = facet.get_unique_nodes()
            pt0, pt1, pt2 = pt0.position, pt1.position, pt2.position

            # Create triangle
            triangle = np.array([pt0, pt1, pt2, pt0])

            # Plot triangle
            ax.add_collection3d(Poly3DCollection([triangle[:-1]], color=colors[color_index % len(colors)], alpha=0.25))
            ax.plot(triangle[:, 0], triangle[:, 1], triangle[:, 2], color=colors[color_index % len(colors)])

            # Calculate and plot normal vector
            vn = facet.calculate_normal()

            center = np.mean(triangle[:-1], axis=0)  # Triangle centroid
            ax.scatter(center[0], center[1], center[2], color='black')

            ax.quiver(center[0], center[1], center[2], vn[0], vn[1], vn[2], length=0.1,
                      color=colors[color_index % len(colors)], label=f"Normal_{i}, f_{facet.sequential_id}")

            # Increment the color index for the next facet
            color_index += 1
        # Set plot limits based on min and max values
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.title('Facets and Normal Vectors Visualization')
        plt.legend()
        plt.show()


    def draw_cell(cell):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw nodes
        for node in cell.get_nodes():  # Assuming a method to get all nodes of a cell
            ax.scatter(node.position[0], node.position[1], node.position[2], color='blue')
            ax.text(node.position[0], node.position[1], node.position[2],
                    f"Node\nID: {node.sequential_id}\nUUID: {node.uuid}")

        # Draw filaments
        for filament in cell.get_filaments():  # Assuming a method to get all filaments of a cell
            x = [filament.node1.position[0], filament.node2.position[0]]
            y = [filament.node1.position[1], filament.node2.position[1]]
            z = [filament.node1.position[2], filament.node2.position[2]]
            ax.plot(x, y, z, color='red')
            mid_point = np.mean([x, y, z], axis=1)
            ax.text(mid_point[0], mid_point[1], mid_point[2],
                    f"Filament\nID: {filament.sequential_id}\nUUID: {filament.uuid}")

        # Draw facets (optional)
        # You might use plot_trisurf or a similar method based on the nodes of the filaments

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.show()

    def draw_cell_ids(self):
        fig, ax = plt.subplots()

        # Define starting positions for the different components
        cell_start = 0.8
        node_start = 0.6
        filament_start = 0.4
        facet_start = 0.2

        # Set y-position for text
        y_spacing = 0.15

        # Draw Cell ID
        ax.text(0.1, cell_start, f"Cell:\nID: {self.sequential_id}\nUUID: {self.uuid}",
                verticalalignment='center', horizontalalignment='left')

        # Draw Node IDs
        ax.text(0.1, node_start, "Nodes:\n" + "\n".join(f"ID: {node.sequential_id}\nUUID: {node.uuid}" for node in self.get_nodes()),
                verticalalignment='center', horizontalalignment='left')

        # Draw Filament IDs
        ax.text(0.1, filament_start, "Filaments:\n" + "\n".join(f"ID: {filament.sequential_id}\nUUID: {filament.uuid}" for filament in self.get_filaments()),
                verticalalignment='center', horizontalalignment='left')

        # Draw Facet IDs
        ax.text(0.1, facet_start, "Facets:\n" + "\n".join(f"ID: {facet.sequential_id}\nUUID: {facet.uuid}" for facet in self.facets),
                verticalalignment='center', horizontalalignment='left')

        # Hide axes
        ax.axis('off')

        plt.show()

    def add_filament(self, node1_id, node2_id, system):
        """
        Add a new filament to the cell.

        Args:
            node1_id (int): The ID of the first node.
            node2_id (int): The ID of the second node.
            system (Systeme): The system to which this filament belongs.
        """
        new_filament = Filament(node1_id, node2_id, self.cell_id)
        self.filament_list.append(new_filament)
        # Additional setup or adjustments can be done here if needed.
    def cell_behaviour(self,longest=False):
        """
        Run all the behaviour functionalities of the epithelium.

        """
        # First test if there is any filament that need to divide
        logger.info(f'Model behaviour ')
        logger.info(f"Cell volume : {self.volume} and cell pressure : {self.pressure}")
        # get all filaments
        Filaments = self.get_filaments()
        concerned_facets = []
        third_node_for_concerned_facet_dict = {}
        num = 0
        # self.is_normals_outward()
        one_time = True
        if longest :
            Filaments =  sorted(Filaments, key=lambda filament: filament.length,reverse = True)
        for filament in Filaments:
            if filament.length > filament.ldivide and one_time:
                logger.superior_info(f"Division {self.division_counter}")
                logger.info(f"Division of : {filament}")
                # get concerned Facets: There should be two
                for facet in self.facets:
                    # logger.info(f"{facet}")
                    if facet.contains_filament(filament):
                        concerned_facets.append(facet)
                        third_node = [node for node in facet.get_unique_nodes() if not filament.made_of_this(node)][0]
                        third_node_for_concerned_facet_dict[facet.sequential_id] = third_node
                        # logger.info(f"Third node for facet:{facet.sequential_id} :{third_node}")
                # logger.info(f" How many concerned facet : {len(concerned_facets)}")
                # logger.info(f"dict for third node :{third_node_for_concerned_facet_dict}")
                # logger.info(f"First concerned facet   :{concerned_facets[0]}")
                # logger.info(f"Second concerned facet  :{concerned_facets[1]}")

                # logger.info(f"Filament {filament.sequential_id} need to be divided!")
                # add new node at filament midpoint
                new_node = Node(position=filament.midpoint())
                # let this node inherit the velocity of the filament
                new_node.velocity = filament.get_velocity()
                # logger.info(f"new_node : {new_node}")
                # create new filament from with old node2 and new_node
                new_Filament = Filament(node1=filament.node2, node2=new_node, cell_id=self.sequential_id)
                new_Filament.increment_age() # because it should have the same age as the original filament \TODO
                new_Filament.increment_age() # because it should have the same age as the original filament \TODO
                # logger.info(f"new filament {new_Filament.sequential_id} age :{new_Filament.age_steps}")
                # repalce the node 2 in the old filament
                filament.replace_node2(new_node)
                # create two new adjacent filaments
                # frist get the concerned third node from concerned_facets
                # redefine the facets
                for i, facet in enumerate(concerned_facets):
                    # get the third node for the concerned facet
                    third_node = third_node_for_concerned_facet_dict[facet.sequential_id]
                    # logger.info(f"concerned facet {facet}")
                    if facet.contains_filament(filament):
                        # logger.info(f"Facet contain a divided filament {filament} !!!")
                        # there should be an unconnected filament in this facet so lets remove it
                        removed_filament,removed_filament_id = facet.detect_disconnected_filament(filament)
                        if removed_filament_id is not None:
                            # logger.info(f"Removed unconnected filament with ID: {removed_filament_id}")
                            new_adj_Filament = Filament(node1=third_node, node2=new_node, cell_id=self.sequential_id)
                            # let's add an adjacent filament to the facet
                            facet.add_filament(new_adj_Filament, removed_filament_id)
                            # logger.info(f"Facet status now : {facet}")
                            new_facet = Facet(filament1=new_Filament,filament2=removed_filament,filament3=new_adj_Filament)
                            # logger.info(f"new_facet: {new_facet}")

                            self.facets.append(new_facet)
                            new_adj_Filament.increment_age()  # because it should have the same age as the original filament \TODO

                        else:
                            raise ValueError("No unconnected filament found.")
                self.division_counter += 1
                one_time = False
                num=num+1
            concerned_facets = []
            # check if all the facets are in the right order
            tetra_center = self.center()
            for facet in self.facets:
                facet.tetrahedron_center = tetra_center
                facet.is_normal_outward()

class Epithelium:
    """
    Definition of a mass-spring system
    node_positions : (x, y, z)
    node_velocities : (vx, vy, vz)
    node_families : family_index in node_properties_list
    elements : node_start, node_end, element_family, lfree
    node_properties : "radius": r, "mass": m, "stiffness": k, "color": (r, g, b)
    element_properties : "radius": r, "linear_mass": ml, "stiffness": EA, "color": (r, g, b)
    locked_nodes_list : node_index, bx, by, bz (0 = free, 1 = locked)
    """
    node_positions = []
    node_velocities = []
    kinetic_energy = []
    node_families = []
    element_list = []
    fixed_element_list = []
    node_properties_list = []
    filament_properties_list = []
    list_blocked_nodes = []
    selected_nodes_list = []
    selected_elements_list = []
    cell_list = []
    contact_nodes_list = []

    def __init__(self):
        """
        Initialization: set default values
        here: elementary system = 1 mass
        """
        # one node
        # self.add_node_family((0., 0., 1.), 0)
        self.cells = []

        # setting the families of nodes and elements
        for i in range(10):
            self.node_properties_list.append({"radius": 1., "mass": 1., "stiffness": 100, "color": (1., 0., 0.)})
            self.filament_properties_list.append({"radius": .5, "linear_mass": 1., "rig_EA": 100, "color": (1., 0., 0.)})
        self.step = None
        self.adhesions = [] # List of adhesions

    @property
    def state(self):
        """
        Property that returns the current state (positions and velocities) of the nodes,
        taking into account any blocked nodes.

        Returns:
            np.array: Current state of the system.
        """
        # logger.info("Getting current state")
        Nodes = self.get_all_nodes()
        Positions = np.array([node.position for node in Nodes])

        # Adjust velocities based on the blocked status of each node
        Velocities = np.array([node.velocity if not node.is_blocked else np.zeros(3) for node in Nodes])

        # Compact formatting of the output
        # logger.info("Node States:")
        # for idx, node in enumerate(Nodes):
        #     logger.info(f"Node {idx}: Pos={Positions[idx]}, Vel={Velocities[idx]}")
        return np.array([Positions, Velocities])

    @state.setter
    def state(self, value):
        """
        Set the current state of the system.

        Args:
            value (np.array): The new state to set.
        """
        # Update the state based on the provided value

        # logger.info("Setting new state")
        Nodes = self.get_all_nodes()
        new_positions, new_velocities = value

        for i, node in enumerate(Nodes):
            # logger.info(f"Updating Node {i}: Pos {node.position} -> {new_positions[i]}, Vel {node.velocity} -> {new_velocities[i]}")
            node.position = new_positions[i]
            node.velocity = new_velocities[i]
    def get_all_nodes(self):
        """
        Returns a list of all nodes in the epithelium.

        Returns:
            list: A list of Node instances from all cells.
        """
        all_nodes = []
        for cell in self.cells:
            all_nodes.extend(cell.get_nodes())

        return all_nodes

    def compute_neighbors(self,threshold = 10):
        """
        Computes the neighbors for each cell within the epithelium based on their spatial proximity.
        Utilizes a KDTree for efficient nearest neighbor search to determine which cells are within a
        certain threshold distance of each other. This method is useful for modeling interactions between
        cells that are physically close to one another.

        Returns:
            dict: A dictionary where each key is the index of a cell within the epithelium's cell list, and
                  the value is a list of indices of neighboring cells within the specified threshold distance.
        """

        # Compute centroids for all cells using your `center` method
        centroids = np.array([cell.center() for cell in self.cells])

        # Build KDTree for efficient neighbor search
        tree = KDTree(centroids)

        # Dictionary to store neighbors for each cell
        cell_neighbors = {}

        # Query neighbors within threshold for each cell
        for i, centroid in enumerate(centroids):
            # indices of neighbors within threshold, including self
            indices = tree.query_ball_point(centroid, r=threshold)
            # Remove self-index
            indices = [ind for ind in indices if ind != i]
            cell_neighbors[i] = indices

        logger.info(f"cell neighbors : {cell_neighbors}")

        return cell_neighbors

    def compute_node_neighbors(self, threshold_distance=2.5):
        """
        Computes the nearest neighbor for each node across all cells within the epithelium based on their spatial proximity,
        excluding nodes that belong to the same cell.

        Args:
            threshold_distance (float): The maximum distance between nodes to consider them as neighbors.

        Returns:
            dict: A dictionary where each key is a node's sequential_id, and the value is the sequential_id
                  of the nearest neighboring node from a different cell, if any.
        """
        node_neighbors = {}

        # Combine all nodes from all cells and get their positions, IDs, and cell IDs
        all_nodes = [(node, cell.sequential_id) for cell in self.cells for node in cell.get_nodes()]
        node_positions = np.array([node[0].position for node in all_nodes])
        node_ids = [node[0].sequential_id for node in all_nodes]
        cell_ids = [cell_id for _, cell_id in all_nodes]

        # Construct KDTree for all nodes
        tree = KDTree(node_positions)

        # Query KDTree for each node to find neighbors within threshold_distance
        for i, node_position in enumerate(node_positions):
            distances, indices = tree.query(node_position, k=len(node_positions))
            nearest_neighbor = None
            for dist, idx in zip(distances[1:], indices[1:]):  # skip the first one because it's the node itself
                if dist > threshold_distance:
                    break
                if cell_ids[idx] != cell_ids[i]:  # check if the node is from a different cell
                    nearest_neighbor = node_ids[idx]
                    break
            if nearest_neighbor:
                node_neighbors[node_ids[i]] = nearest_neighbor

        logger.info(f"node_neighbors: {node_neighbors}")

        return node_neighbors

    def create_an_eptm_of_a_growing_cells(self):
        """
        Creates an epithelial tissue model with one growing cells.
        """
        cell1 = Cell()
        cell1.redefine_volume0()
        self.cells.append(cell1)


    def create_an_eptm_of_two_growing_cells(self):
        """
        Creates an epithelial tissue model with two growing cells.
        """
        cell1 = Cell()
        cell2 = Cell()

        translation_vector = [0, 3, 0]  # Translate +20 on the z-axis
        cell2.translate_cell(translation_vector)

        # translate out of the plane
        cell1.translate_cell([0, 0, 5])
        cell2.translate_cell([0, 0, 5])

        self.cells.append(cell1)
        self.cells.append(cell2)
        for cell in self.cells:
            cell.redefine_volume0()

    def create_an_eptm_of_five_growing_cells(self):
        """
        Creates an epithelial tissue model with five growing cells.
        """
        # Create five cells
        cell1 = Cell()
        cell2 = Cell()
        cell3 = Cell()
        cell4 = Cell()
        cell5 = Cell()

        # Define translation vectors for the cells to avoid overlap and arrange them spatially
        translation_vectors = [
            [0, 2, 0],  # Cell 2
            [2, 0, 0],  # Cell 2
            [0, -2, 0],  # Cell 4
            [-2, 0, 0]  # Cell 5
        ]

        # Translate the cells according to the defined vectors
        cell2.translate_cell(translation_vectors[0])
        cell3.translate_cell(translation_vectors[1])
        cell4.translate_cell(translation_vectors[2])
        cell5.translate_cell(translation_vectors[3])

        # Translate all cells out of the plane
        for cell in [cell1, cell2, cell3, cell4, cell5]:
            cell.translate_cell([0, 0, 5])

        # Append the cells to the model
        self.cells.append(cell1)
        self.cells.append(cell2)
        self.cells.append(cell3)
        self.cells.append(cell4)
        self.cells.append(cell5)

    def create_an_eptm_of_nine_growing_cells(self):
        """
        Creates an epithelial tissue model with ten growing cells.
        """
        # Create ten cells
        cells = [Cell() for _ in range(9)]

        # Define translation vectors for the cells to avoid overlap and arrange them spatially
        translation_vectors = [
            [0, 2, 0],  # Cell 2
            [2, 0, 0],  # Cell 3
            [0, -2, 0],  # Cell 4
            [-2, 0, 0],  # Cell 5
            [2, 2, 0],  # Cell 6
            [-2, -2, 0],  # Cell 7
            [-2, 2, 0],  # Cell 8
            [2, -2, 0],  # Cell 9
        ]

        # Translate the cells according to the defined vectors
        for i, translation_vector in enumerate(translation_vectors):
            cells[i + 1].translate_cell(translation_vector)

        # Translate all cells out of the plane
        for cell in cells:
            cell.translate_cell([0, 0, 5])

        # Append the cells to the model
        self.cells.extend(cells)

    def get_all_filaments(self):
        """
        Returns a list of all filaments in the epithelium.

        Returns:
            list: A list of Filament instances from all cells.
        """
        all_filaments = []
        for cell in self.cells:
            all_filaments.extend(cell.get_filaments())  # Call the get_filaments method of each cell

        return all_filaments

    def get_all_facets(self):
        """
        Returns a list of all facets in the epithelium.

        Returns:
            list: A list of Facet instances from all cells.
        """
        all_facets = []
        for cell in self.cells:
            all_facets.extend(cell.facets)  # Directly access the facets attribute of each cell

        return all_facets

    def dynamic_evolution(self, t, dt):
        """
        Run the dynamic evolution of the epithelium
        """
        # current_state = self.state
        # next_state = self.RKOneD(t, dt)
        next_state = self.EulerOneD(self.state,t, dt)

        self.state = next_state

        # Increment the age of all filaments in each cell by one simulation step
        for filament in self.get_all_filaments():
            filament.increment_age()

        # # resetting volume 0 for cells
        # for cell in self.cells:
        #     cell.volume0 += 0.2*cell.volume

    def RKOneD(self, t, dt):
        """
        RungeKutta order 4. Requires states of type np.array.
        """
        x = self.state.copy()
        # logger.info("First call ")
        k1 = dt * self.derive_state(x, t)
        # logger.info(f"x before second call : {x}")
        # logger.info("Second call ")
        k2 = dt * self.derive_state(x + k1 / 2.0, t)
        # logger.info("Third call ")
        k3 = dt * self.derive_state(x + k2 / 2.0, t)
        # logger.info("Last call ")
        k4 = dt * self.derive_state(x + k3, t)
        etat_tpdt = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return etat_tpdt

    def EulerOneD(self, x, t, dt):
        """
        Euler 1D
        """
        etat_tpdt = x + dt * self.derive_state(x, t)
        return etat_tpdt

    def derive_state(self, state, t):
        """
        Derives the state of the system, given in the form (position, velocity)
        In dynamics, the derived system is (velocity, acceleration), with acc = F/m
        Hence, it boils down to calculating the forces applied to the nodes
        """
        logger.superior_info("Derive state")
        # for cell in self.cells:
        #     cell.plot_facets_with_normals()
        state2 = state.copy() * 0.
        nb_nodes = len(state[0])
        logger.info(f"Current number of nodes : {nb_nodes}")
        logger.debug("#################################### Derive state ##############################################" )

        Nodes = self.get_all_nodes()
        Filaments = self.get_all_filaments()
        gravity_force = 0
        viscous_damping_force =0
        ground_contact_force = 0
        for node in Nodes:
            node.reset_forces()
            ground_contact_force = node.ground_contact_force()
            node.forces += ground_contact_force
        for i, ndf in enumerate(Nodes):
            logger.dont_debug(f"GF forces : {ndf.forces}")
        for node in Nodes:
            gravity_force = node.gravity_force(params["gravity"])
            viscous_damping_force = node.viscous_damping_force(params["medium_viscosity"])
            # Summing up the forces for each node
            node.forces +=  gravity_force + viscous_damping_force
        for i, ndf in enumerate(Nodes):
            logger.dont_debug(f"GGV forces : {ndf.forces}")
        # Calculate and assign filament forces to each node
        Fil_forces = []
        for filament in Filaments:
            force = filament.elastic_force()
            filament.node1.forces += force
            filament.node2.forces -= force
            Fil_forces.append(force)
            Fil_forces.append(force)
        Fil_forces = []
        for i, ndf in enumerate(Nodes):
            logger.dont_debug(f"GGV+E forces : {ndf.forces}")
        # Example usage
        # self.visualize_filament_forces(Nodes, Filaments, plot_resultant_forces=True)

        for cell in self.cells:
            # cell.calculate_volume()
            cell.calculate_pressure()
            cell.apply_pressure_to_nodes()
            logger.debug(f"Cell volume after calc : {cell.volume} ,cell volume0 {cell.volume0} and cell pressure: {cell.pressure}")

        # add pressure forces to nodes forces
        for i,node in enumerate(Nodes):
            node.add_force(node.pressure_forces)
        for i, ndf in enumerate(Nodes):
            logger.dont_debug(f"GGV+E+P forces : {ndf.forces}")

        # Adhesion Forces
        self.update_adhesion()
        tcc0 = time.time()
        tcc1 = time.time()
        global delta_time_ccc
        delta_time_ccc = tcc1 - tcc0
        ########################################################################
        # calculation of the derivative state
        mass_nodes = np.array([node.properties['mass'] for node in Nodes])
        forces_nodes = np.array([node.forces for node in Nodes])
        for i, ndf in enumerate(Nodes):
            logger.dont_debug(f"All forces : {ndf.forces}")
        # Reshape mass_nodes to (4, 1) for element-wise division with forces_nodes
        mass_nodes = mass_nodes.reshape(-1, 1)
        # logger.debug(f"Mass nodes : {mass_nodes}")
        state2[1] = forces_nodes/mass_nodes
        state2[0] = state[1].copy()
        logger.info(f"####################################################################")

        # logger.debug(f"State 2 : \n {state2}")

        return state2


    def update_adhesion(self):
        self.compute_neighbors()
        node_neighbors = self.compute_node_neighbors()
        nodes = self.get_all_nodes()
        nodes_dict = {}
        print(f"Adhesions Number : {len(self.adhesions)}")

        if not self.adhesions:
            for node in nodes:
                nodes_dict[node.sequential_id] = node
            # define new filament as adhesion filamnet
            for idx in node_neighbors.items():
                ad_fil = Filament(node1=nodes_dict[idx[0]],node2=nodes_dict[idx[1]],cell_id=None,type="adhesion")
                if ad_fil is not None:
                    self.adhesions.append(ad_fil)
            print(f"Adhesions Number : {len(self.adhesions)}")
            for ad in self.adhesions:
                print(ad)
    def pickle_self(self, SAVE_DIR=None, name=None, prune_adhesions=True):
        """Pickles and saves an instance of this class in its current state.

        :param SAVE_DIR: (Default value = None)  Save location.
        :type SAVE_DIR: string
        :param name:  (Default value = None)  Filename
        :type name: string
        :param prune_adhesions: (Default value = True)  Remove fast adhesions and cell-stored adhesions before saving (recommended for space).
        :type prune_adhesions : bool

        """

        print("Saving Epithelium objects", object(), 1)

        # if prune_adhesions:
        #     for cell in self.cells:
        #         cell.prune_adhesion_data()
        #     if self.boundary_bc != 'elastic':
        #         self.reference_boundary_adhesions = []

        if SAVE_DIR == None:
            SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pickled_toygl_tissues'))
            print(f"Save dir : {SAVE_DIR}")
        # Make sure the directory exists
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        # Filename
        if name == None:
            name = 'Epithelium_cell0_uuid' + str(self.cells[0].uuid) + '_' + '_pressure_' \
                   + str(self.cells[0].pressure)
        # saveloc
        saveloc = SAVE_DIR + '/' + name
        # Pickle
        with open(saveloc, 'wb') as s:
            dill.dump(self, s)
    def model_eptm_behaviour(self,t):
        """
        Args :
            t (float): simulation time step

        Run all the behaviour functionalities of the epithelium.
        """
        logger.superior_info(f"###Time {t} ###")
        t = np.round(t,3)
        if abs(t % 0.1) < 1e-5:  # 1e-5 is a small number to account for floating-point arithmetic issues
            # Perform the action you want to do every time time variable increases by 1
            logger.superior_info(f"### Resetting division_counter ###")
            for cell in self.cells:
                cell.division_counter = 0
        if abs(t % 1) < 1e-5:  # 1e-5 is a small number to account for floating-point arithmetic issues
            cell.volume0 += 50 * dt * 1
        # Increase cell intital volume :volume0
        for cell in self.cells:
            F = len(cell.get_filaments())
            logger.info(f"le volume de la cellule en cours est {cell.volume} et F = {F} ")
            if cell.pressure < 400 and cell.volume0 < 75 :  # garde fou
                logger.debug(f"Increment : {50 * dt * 1}")
                cell.volume0 += 50 * dt * 1
            # if cell.pressure < 0:
                #     raise ValueError(f"Pressure is negative !!!!, vol0 is {cell.volume0} and vol is : {cell.volume}")
            if cell.division_counter < 1 and F < 25:# limit to 20 division
                if len(self.cells) == 1:
                    cell.cell_behaviour()
            if cell.division_counter < 1 and F < 100 and step > 880 :
                if len(self.cells) == 1:
                    cell.cell_behaviour(longest=True)

            else:
                logger.superior_info(f"### Limit of {cell.division_counter} division has been reached ###")
            # else:
            #     raise ValueError("Threshold reached!!!!")



    # functions for debbug
    def visualize_filament_forces(self,Nodes, Filaments, plot_resultant_forces=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Define colors for each filament
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']

        # Extract node positions
        positions = np.array([node.position for node in Nodes])

        # Plot nodes
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='black', s=20)

        # Plot filaments and normalized force vectors
        for i, filament in enumerate(Filaments):
            force = filament.elastic_force()
            normalized_force = force / np.linalg.norm(force) if np.linalg.norm(force) != 0 else np.zeros_like(force)
            color = colors[i % len(colors)]  # Cycle through the defined colors

            # Node positions
            pos1 = filament.node1.position
            pos2 = filament.node2.position

            # Plot filament as a black line
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], [pos1[2], pos2[2]], color='black')

            # Plot normalized force vectors at each node position
            ax.quiver(pos1[0], pos1[1], pos1[2], normalized_force[0], normalized_force[1], normalized_force[2],
                      length=0.1, color=color)
            ax.quiver(pos2[0], pos2[1], pos2[2], -normalized_force[0], -normalized_force[1], -normalized_force[2],
                      length=0.1, color=color,linestyle ='--')

            # Label filament with its ID in the force color
            mid_point = (pos1 + pos2) / 2
            ax.text(mid_point[0], mid_point[1], mid_point[2], f'Filament {i}', color=color)

        # Optionally plot normalized resultant forces at each node
        if plot_resultant_forces:
            for node in Nodes:
                res_force = node.forces
                normalized_res_force = res_force / np.linalg.norm(res_force) if np.linalg.norm(
                    res_force) != 0 else np.zeros_like(res_force)
                ax.quiver(node.position[0], node.position[1], node.position[2], normalized_res_force[0],
                          normalized_res_force[1], normalized_res_force[2], length=0.1, color='orange')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.title('Filament Forces Visualization')
        plt.show()
    def display_stats(self):
        """
        Displays statistics on the current state
        """
        # position of the selected node
        if len(self.selected_nodes_list) == 1:
            no_sel = self.selected_nodes_list[0]
            pt_no_sel = self.node_positions[no_sel]
            logger.info("Node ", no_sel, pt_no_sel)

        # L, lfree and force in the selected element
        if len(self.selected_elements_list) == 1:
            el_sel = self.selected_elements_list[0]
            el = self.element_list[el_sel]
            pt1 = self.node_positions[el[0]]
            pt2 = self.node_positions[el[1]]
            fam = el[2]
            lfree = el[3]
            car = self.filament_properties_list[fam]
            l = norm(np.array(pt2) - np.array(pt1))
            n = car["rig_EA"] * (l - lfree) / lfree
            logger.info("Element ", el_sel, " - L = ", l, " - lfree = ", lfree, " - T = ", n)

        # pressions des cellules
        for i_cell, celli in enumerate(self.cell_list):
            logger.info(f" - cell {i_cell} - pressure={celli.pressure} - volume={celli.volume} - volume0={celli.volume0}")


# ------------------------------------------------------------------------------
def draw_cylinder(pt1, pt2, rayon1, rayon2):
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


def draw_sphere(pos, radius):
    """
    Draw a sphere at the indicated position, of the indicated radius
    (10 subdivisions by default)

    """
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glutSolidSphere(radius, 10, 5)
    glPopMatrix()

def draw_arrow(start_point, end_point, arrow_radius=0.05, head_length=0.2, head_radius=0.1, color=(1.0, 1.0, 1.0)):
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
    vector = np.array(end_point) - np.array(start_point)
    length = np.linalg.norm(vector)

    # Normalize the vector
    if length != 0:
        direction = vector / length
    else:
        direction = vector

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


def draw_reference_frame():
    return
    draw_arrow(start_point=(0, 0, 0), end_point=(7, 0, 0), color=(1, 0, 0))  # X-axis (Red)
    draw_arrow(start_point=(0, 0, 0), end_point=(0, 7, 0), color=(0, 1, 0))  # Y-axis (Green)
    draw_arrow(start_point=(0, 0, 0), end_point=(0, 0, 7), color=(0, 0, 1))  # Z-axis (Blue)


def draw_plane():
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

def display():
    # preparation timing
    global previousTime, frameCounter, frameCounterTimer, envlist
    currentTime = time.time()
    deltaTime = (currentTime - previousTime)

    # effacement + placement camera
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear buffers
    glEnable(GL_DEPTH_TEST)  # Enable depth testing
    glEnable(GL_BLEND)  # Enable blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Set blend function

    setCamera()

    #########################################
    # Draw nodes, filaments, facets, reference frame, and plane
    for node in mon_sys.get_all_nodes():
        node.draw(selected=True, in_contact=False)

    for filament in mon_sys.get_all_filaments():
        filament.draw(selected=False)

    for facet in mon_sys.get_all_facets():
        facet.draw(color=(0, 0, 1, 0.5))  # blue color with 50% transparency
    if mon_sys.adhesions:
        for adhesion in mon_sys.adhesions:
            print(f"adhesion : {adhesion}")
            adhesion.draw(selected=True)
    draw_reference_frame()
    draw_plane()

    #########################################
    # basculement buffer : affichage
    glutSwapBuffers()

    # timing
    previousTime = currentTime
    frameCounter += 1
    if currentTime - frameCounterTimer > 1:
        # affichage toutes les secondes
        logger.info(f"FPS: {frameCounter}")
        logger.info(f"t = {t} - delta_ccc = {delta_time_ccc}")
        frameCounter = 0
        frameCounterTimer = currentTime
        mon_sys.display_stats()


def resize(width, height):
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



def setCamera():
    global azim, elev, distance, origine

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # Apply transformations based on updated azim and elev values
    glTranslatef(0., 0., -distance)
    # Set elevation and azimuth to align with the XZ plane
    # elev = 80 # Rotate 90 degrees around the X-axis to switch from XY to XZ plane
    # azim = 0 # No rotation around the Y-axis needed for this plane
    glRotatef(elev, 1, 0., 0.)   # Rotate around the x-axis for elevation
    glRotatef(azim, 0., 1, 0.)   # Rotate around the y-axis for azimuth
    glTranslatef(-origine[0], -origine[1], -origine[2])


def motion(x, y):
    """
    Mouse movement management
    """
    global mouse_orig, rotate, zoom, translation, distance, elev, azim

    if rotate:
        azim += 0.2 * (x - mouse_orig[0])
        elev += 0.2 * (y - mouse_orig[1])
        azim = azim % 360.0
        elev = elev % 360.0
        mouse_orig[0] = x
        mouse_orig[1] = y
        print(f"Rotating: azim={azim}, elev={elev}")
        glutPostRedisplay()

    elif zoom:
        distance += 0.01 * (y - mouse_orig[1])
        mouse_orig[1] = y
        print(f"Zooming: distance={distance}")
        glutPostRedisplay()

    elif translation:
        v = [np.sin(azim * np.pi / 180), np.cos(azim * np.pi / 180)]
        vp = [v[1], -v[0]]
        dx = (x - mouse_orig[0])
        dy = (y - mouse_orig[1])
        origine[0] += 0.1 * (dx * vp[0] - dy * v[0])
        origine[1] += 0.1 * (dx * vp[1] - dy * v[1])
        mouse_orig[0] = x
        mouse_orig[1] = y
        print(f"Translating: origine={origine}")
        glutPostRedisplay()

def mouse(button, state, x, y):
    """
    Mouse management
    """
    global mouse_orig, rotate, zoom, translation

    if button == GLUT_LEFT_BUTTON:
        if state == GLUT_DOWN:
            if glutGetModifiers() == GLUT_ACTIVE_CTRL:
                print("Left button with CTRL pressed: Zoom mode")
                mouse_orig[1] = y
                zoom = True
                rotate = False
                translation = False
            else:
                print("Left button pressed: Rotate mode")
                mouse_orig[0] = x
                mouse_orig[1] = y
                rotate = True
                zoom = False
                translation = False
        else:
            print("Left button released")
            rotate = False

    elif button == GLUT_RIGHT_BUTTON:
        if state == GLUT_DOWN:
            if glutGetModifiers() == GLUT_ACTIVE_CTRL:
                print("Right button with CTRL pressed: Translation mode")
                mouse_orig[0] = x
                mouse_orig[1] = y
                translation = True
                rotate = False
                zoom = False
            else:
                print("Right button pressed: Zoom mode")
                mouse_orig[1] = y
                zoom = True
                rotate = False
                translation = False
        else:
            print("Right button released")
            zoom = False
            translation = False


is_paused = False

def toggle_pause():
    global is_paused
    is_paused = not is_paused

def key(key, x, y):
    global is_paused
    if key == b'p':  # Assuming pressing 'p' should toggle pause
        toggle_pause()

# Global variable to keep track of the number of evolution steps
step = 0
evolution_limit = 9000  # Limit the number of dynamic evolution steps to 3

# Set the delay duration in seconds (e.g., 0.5 seconds)
frame_delay = 0# 0.00001

def animation():
    save_dir = '/home/ardati/PycharmProjects/pickled_toygl_tissues'
    global t, dt, t_div_elem, step,last_interval_index
    # last_interval_index =
    if is_paused:
        return
    else:
        if step < evolution_limit:
            for i in range(1):
                t += dt  # Advance time
                logger.superior_info(f"At step : {step}")
                logger.debug(f"At step : {step}")

                mon_sys.state
                mon_sys.dynamic_evolution(t, dt)  # Dynamic evolution of the epithelium
                # mon_sys.model_eptm_behaviour(t)
                step += 1  # Increment the evolution counter
                # Additional evolution functions can be added here
                # Save the advancement
                mon_sys.step = step
                mon_sys.pickle_self(SAVE_DIR=save_dir,name=f"step_{step}")

        else:
            logger.info("Evolution limit reached.")
            # mon_sys.cell_behaviour(t)
        display()

        # Introduce a delay between frames
        time.sleep(frame_delay)




if __name__ == '__main__':
    ######################################
    ### programme principal
    ####################################
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
    glutDisplayFunc(display)
    glutIdleFunc(animation)
    glutReshapeFunc(resize)
    # glutIgnoreKeyRepeat(1)
    glutKeyboardFunc(key)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    Save_dir = "/home/ardati/Data_ToyGL/"
    sim_name = "Test_adhesion"
    Save_dir = Save_dir + sim_name
    print(f"Save dir : {Save_dir}")
    # Intialise an epthelium
    mon_sys = Epithelium()
    Recup = False
    if Recup:
        print(f"## Loading a toygl tissue ##...  ")
        Save_dir = "/home/ardati/PycharmProjects/pickled_toygl_tissues"
        tissues = humansorted(os.listdir(Save_dir))
        tissu_name = tissues[-1]
        # or choose a specific tissue to loop from
        tissu_name = 'step_880'

        print(f"tissues : {tissues}")
        open_dir = os.path.join(f"{Save_dir}", f"{tissu_name}")
        with open(open_dir, 'rb') as s:
            mon_sys = dill.load(s)
            print(f"Successfully loaded Relaxed Voronoi tissue: {tissu_name}")

        print(f" mon_sys.step : {mon_sys.step}")
        step = mon_sys.step
    else :
        Gcells = mon_sys.create_an_eptm_of_nine_growing_cells()
        # Gcells = mon_sys.create_an_eptm_of_five_growing_cells()
        # Gcells = mon_sys.create_an_eptm_of_two_growing_cells()
        # Gcells = mon_sys.create_an_eptm_of_a_growing_cells()


    # Gcells = mon_sys.create_an_eptm_of_two_growing_cells()


    # launching Glut
    glutMainLoop()
