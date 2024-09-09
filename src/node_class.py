# src/node_class.py
import numpy as np
import uuid

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
        self._position_changed = False  # Track whether the position has changed

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = np.array(value)
        self._position_changed = True  # Mark the position as changed

    def has_position_changed(self):
        return self._position_changed

    def reset_position_changed(self):
        self._position_changed = False

    def __sub__(self, other_node):
        """
        Subtract the position of another node from this node's position.

        Args:
            other_node (Node): The node to subtract.

        Returns:
            np.ndarray: The resulting vector from the subtraction.
        """
        if isinstance(other_node, Node):
            return self.position - other_node.position
        else:
            raise TypeError("Subtraction can only be performed between Node instances.")

    def __repr__(self):
        return (f"Node {self.sequential_id}")

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
        dz = 0
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