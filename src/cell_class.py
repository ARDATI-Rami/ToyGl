from src.facet_class import Facet
from src.filament_class import Filament
from src.node_class import Node
from src.logging_utils import logger
from src.math_utils import *


import uuid
import hashlib
import numpy as np

from scipy.spatial import ConvexHull


class Cell:
    """
    A class to describe a 3D cell
    """
    liste_tri3 = []  # list of 3 nodes ids describing a facet, facing outside
    volume = 1.  # calculé à chaque pas de temps
    dp_dv = 2.  # rigidity
    _next_id = 0
    pressure = 0
    target_volume = 50

    def __init__(self):
        """
        Init cell with a single tetraedra
         """

        self.initialize_basic_cell()
        combined_uuids = ''.join(str(facet.uuid) for facet in self.facets)
        self.uuid = uuid.UUID(hashlib.md5(combined_uuids.encode()).hexdigest())
        self.sequential_id = Cell._next_id
        Cell._next_id += 1
        self.nodes = self.get_nodes()
        self.volume0 = self.calculate_volume()

        self.division_counter = 0
        self.is_normals_outward()
        logger.dont_debug(f"Volume 0 is : {self.volume0}")

    def create_icosphere(self,depth=2):
        phi = (1 + 5 ** 0.5) / 2  # Golden ratio

        # Initialize vertices
        vertices = [
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
        ]

        # Normalize all vertices
        vertices = [normalize(v) for v in vertices]

        # Initialize faces
        faces = [
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
        ]

        def subdivide(vertices, faces, depth):
            vertices = list(vertices)
            for _ in range(depth):
                new_faces = []
                midpoint_cache = {}

                def cached_midpoint(v1, v2):
                    key = tuple(sorted((v1, v2)))
                    if key not in midpoint_cache:
                        midpoint_cache[key] = len(vertices)
                        mid = normalize(midpoint(vertices[v1], vertices[v2]))
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

            return vertices, faces

        vertices, faces = subdivide(vertices, faces, depth)

        return vertices, faces
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
            vertices, faces = self.create_icosphere(depth)

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


    def translate_cell(self, translation_vector):
        """
        Translates the entire cell by a given vector.

        Args:
            translation_vector (list or np.array): A vector specifying the translation amount in each dimension.
        """
        for node in self.nodes:
            node.position += np.array(translation_vector)
            logger.info(f"Node {node.sequential_id} translated to new position: {node.position}")

    def get_nodes(self):
        """
        Returns a list of all unique nodes in the cell.

        Returns:
            list: A list of unique Node instances.
        """
        unique_nodes = {}

        for filament in self.get_filaments():
            if filament.node1.uuid not in unique_nodes:
                unique_nodes[filament.node1.uuid] = filament.node1
            if filament.node2.uuid not in unique_nodes:
                unique_nodes[filament.node2.uuid] = filament.node2
        self.nodes = list(unique_nodes.values())

        return self.nodes

    def get_filaments(self):
        """
        Returns a list of all unique filaments in the cell.

        Returns:
            list: A list of unique Filament instances.
        """
        unique_filaments = {}

        for facet in self.facets:
            for filament in facet.filaments:
                if filament.uuid not in unique_filaments:
                    unique_filaments[filament.uuid] = filament

        return list(unique_filaments.values())

    def center(self):
        """
        Calculate and return the centroid of the cell.

        Returns:
            numpy.array: The centroid of the cell.
        """
        if not self.nodes:
            return np.array([0, 0, 0])  # Default center if no nodes

        positions = np.array([node.position for node in self.nodes])
        centroid = np.mean(positions, axis=0)
        return centroid

    def is_normals_outward(self):
        tetrahedron_center = self.center()
        for facet in self.facets:
            # add the center of the tetrahedron to the facet
            facet.tetrahedron_center = tetrahedron_center
            # Assuming calculate_normal is a method that calculates the normal of a facet
            facet._calculate_normal()
            normal = facet.normal

            # Choose a point on the facet (e.g., the position of the first node of the first filament)
            point_on_facet = facet.filaments[0].node1.position

            # Calculate the vector from the center to the point on the facet
            center_to_point_vector = point_on_facet - tetrahedron_center

            # Calculate the dot product
            if dot_product(normal, center_to_point_vector) > 0:
                logger.info(f"For Facet : {facet.sequential_id} the normal was not outward, Reversing order ...")
                facet.reverse_filaments()


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
        points = np.array([node.position for node in self.nodes])
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
        # Calculate the pressure
        self.pressure = (self.volume0 - self.volume)*self.dp_dv
        logger.dont_debug(f"New pressure : {self.pressure}")

    def apply_pressure_to_nodes(self):
        """
        Applies pressure forces to the nodes of the cell based on its facets.
        """

        for facet in self.facets:
            # Calculate the normal vector of the facet
            vn = facet.normal
            # logger.info(f"vn : {vn}")

            # Calculate the force due to pressure on the facet
            force = [self.pressure * vn[i] * 0.5 for i in range(3)]
            # logger.info(f"v_force : {force} and 1/3 v_force : {force / 3}")

            # Distribute the force to each node of the facet
            unique_nodes = facet.get_unique_nodes()
            for node in unique_nodes:
                node.pressure_forces = [node.pressure_forces[i] + force[i] / 3 for i in range(3)]  # Assuming equal distribution to each node

    def grow_to_target_volume(self):
        """
        Grow the cell to reach the target volume.
        """
        dv = 0.5
        if self.volume < self.target_volume:
            logger.dont_debug("Increasing volume to target volume")
            self.volume += dv
    def plot_facets_with_normals(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Get all nodes and their positions
        positions = np.array([node.position for node in self.nodes])

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
            vn = facet.normal

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


    def draw_cell(self,cell):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw nodes
        for node in cell.nodes:  # Assuming a method to get all nodes of a cell
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
        ax.text(0.1, node_start, "Nodes:\n" + "\n".join(f"ID: {node.sequential_id}\nUUID: {node.uuid}" for node in self.nodes),
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



    def cell_behaviour(self, longest=True, one_division_per_step=True):
        """
        Cell Behaviour includes:
        1- Filament Division:
            For each dividing filament, there are at least two concerned facets (facets that contain this filament).
            The filament made from two nodes will divide in half:
                - The midpoint is called the new node.
                - The third node is the node that is neither of them in a given facet.
            The filament that divides will be removed and replaced with 4 new filaments: d1, d2, d3, and d4.

            It is important to arrange the new filaments in the right order so the normal of those facets
            will have the same direction as the concerned facet.
        2- Filament Death
        3- Filament Constriction in an actin ring
        """
        logger.info(f'Cell {self.sequential_id} behaviour')

        # Get all filaments and sort if required
        filaments = self.get_filaments()
        if longest:
            filaments.sort(key=lambda filament: filament.length, reverse=True)

        for filament in filaments:
            if filament.length > filament.ldivide and one_division_per_step:
                logger.superior_info(f"Division counter {self.division_counter}")
                self.division_counter += 1
                logger.info(f"Division of {filament}")

                # Find concerned facets and third nodes
                concerned_facets = []
                third_nodes = []
                all_concerned_filaments = []

                for facet in self.facets:
                    if facet.contains_filament(filament):
                        concerned_facets.append(facet)
                        all_concerned_filaments.extend(facet.filaments)
                        third_node = [node for node in facet.get_unique_nodes() if not filament.composed_of(node)][0]
                        third_nodes.append(third_node)

                # Add new node at filament midpoint and inherit velocity
                new_node = Node(position=filament.midpoint())
                new_node.velocity = filament.get_velocity()

                # Define vectors based on the node positions
                v1 = new_node - filament.node1
                v2 = filament.node2 - new_node
                v3 = new_node - third_nodes[0]
                v4 = third_nodes[1] - new_node

                # Check if concerned_facets_normals[0] is in the same direction as v1 cross v3
                same_direction_v1_v3 = is_same_direction(v1, v3, concerned_facets[0].normal)

                # Create 4 new filaments based on direction
                d1 = Filament(node1=filament.node1, node2=new_node, cell_id=self.sequential_id)
                d2 = Filament(node1=new_node, node2=filament.node2, cell_id=self.sequential_id)
                if same_direction_v1_v3:
                    d3 = Filament(node1=third_nodes[0], node2=new_node, cell_id=self.sequential_id)
                    d4 = Filament(node1=new_node, node2=third_nodes[1], cell_id=self.sequential_id)
                else:
                    d3 = Filament(node1=new_node, node2=third_nodes[0], cell_id=self.sequential_id)
                    d4 = Filament(node1=third_nodes[1], node2=new_node, cell_id=self.sequential_id)

                # Find adjacent filaments for the new facets
                adf1 = next(fil for fil in all_concerned_filaments if fil.composed_of(third_nodes[0], filament.node1))
                adf2 = next(fil for fil in all_concerned_filaments if fil.composed_of(third_nodes[1], filament.node1))
                adf3 = next(fil for fil in all_concerned_filaments if fil.composed_of(third_nodes[0], filament.node2))
                adf4 = next(fil for fil in all_concerned_filaments if fil.composed_of(third_nodes[1], filament.node2))

                # Define and add new facets
                f1 = Facet(filament1=d1, filament2=d3, filament3=adf1)
                f2 = Facet(filament1=d1, filament2=d4, filament3=adf2)
                f3 = Facet(filament1=d2, filament2=d3, filament3=adf3)
                f4 = Facet(filament1=d2, filament2=d4, filament3=adf4)
                self.facets.extend([f1, f2, f3, f4])

                # Remove the old concerned facets
                for facet in concerned_facets:
                    self.facets.remove(facet)

                # Stop further divisions if only one division is allowed per step
                one_division_per_step = False

