# src/filament_class.py

from src.node_class import Node
from src.logging_utils import logger
from src.math_utils import *

import uuid
import hashlib
import numpy as np


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
        return {'radius': 0.05, 'linear_mass': 0.2, 'rig_EA': 8, 'color': (0.0, 1.0, 0.0)}

    def __new__(cls, node1, node2, cell_id, properties=None, type='Structure'):
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
        # Sort node IDs to create a unique identifier
        node_ids = sorted([str(node1.sequential_id), str(node2.sequential_id)])
        combined_ids = '-'.join(node_ids)


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

        self.ldivide = 1.5 # threshold length for filament division
        self.age_steps = 0  # Initialize age in simulation steps
        self.type = type

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
        return norm(self.node2 - self.node1)

    @property
    def vector(self):
        """Calculate the current vector of the filament."""
        return np.array(self.node2- self.node1)

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

    def composed_of(self, *nodes):
        """
        Check if the filament is made of the given node(s).

        Args:
            nodes (Node): One or two nodes to check.

        Returns:
            bool:
                - True if the filament is made of the single node.
                - True if the filament is made of both nodes when two are provided.
                - False otherwise.
        """

        if len(nodes) == 1:
            # Single node case
            result = nodes[0] == self.node1 or nodes[0] == self.node2
            return result
        elif len(nodes) == 2:
            # Two nodes case
            nodes_set = {self.node1, self.node2}
            result = set(nodes) == nodes_set
            return result
        else:
            raise ValueError("This method accepts only one or two nodes.")

    def flip_in_place(self):
        """
        Flip the filament by reversing the order of its nodes, modifying the object in place.
        """
        # Swap the nodes

        self.node1, self.node2 = self.node2, self.node1

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
        EA = self.properties["rig_EA"] if self.type =='cell' else -1*self.properties["rig_EA"]   # Stiffness of the filament
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
