from src.filament_class import Filament
from src.math_utils import *

import uuid
import hashlib
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
        self._calculate_normal()
        self.is_normal_outward()




    def __repr__(self):
        """
        Return a string representation of the Facet object using the existing __repr__ method of Filaments.
        """
        facet_info = f"Facet {self.sequential_id} details:\n"
        filaments_info = "\n".join([f"{idx+1}: {filament}" for idx, filament in enumerate(self.filaments)])
        return f"{facet_info}{filaments_info}"


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
                normal = self._calculate_normal()

                # Choose a point on the self (e.g., the position of the first node of the first filament)
                point_on_self = self.filaments[0].node1.position
                # Calculate the vector from the center to the point on the self
                center_to_point_vector = point_on_self - self.tetrahedron_center

                # Calculate the dot product
                dot_product = dot_product(normal, center_to_point_vector)

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


    def _calculate_normal(self):
        """Calculate the normal vector of a facet defined by two filaments."""
        self.normal = normalize(cross_product(self.filament1.vector, self.filament2.vector))

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

        # The area of the triangle is half the magnitude of the cross product
        area = 0.5 * normalize(cross_product(v1, v2))
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

