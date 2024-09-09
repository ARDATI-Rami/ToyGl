# src/eptm_class.py
from src.cell_class import Cell
from src.filament_class import Filament

from src.logging_utils import logger

from src.math_utils import *


import os
import shutil
import numpy as np
import pyvista as pv

from scipy.spatial import KDTree


flags = {"dynamic_relaxation": False,
         "cell-cell_contact": True, 'node-node_contact': False}
params = {"gravity": -200*0, "medium_viscosity": 0}


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
        self.step = 0

        # setting the families of nodes and elements
        for i in range(10):
            self.node_properties_list.append({"radius": 1., "mass": 1., "stiffness": 100, "color": (1., 0., 0.)})
            self.filament_properties_list.append(
                {"radius": .5, "linear_mass": 1., "rig_EA": 100, "color": (1., 0., 0.)})
        self.nodes = []
        self.filaments = []
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
        self.nodes = self.get_all_nodes()
        Positions = np.array([node.position for node in self.nodes])

        # Adjust velocities based on the blocked status of each node
        Velocities = np.array([node.velocity if not node.is_blocked else np.zeros(3) for node in self.nodes])

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

        new_positions, new_velocities = value

        for i, node in enumerate(self.nodes):
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

    def update_adhesion(self):
        """
        Update adhesion by calculating neighbors and applying adhesion forces.
        """
        all_nodes = [(node, cell.sequential_id) for cell in self.cells for node in cell.nodes]
        node_positions = np.array([node[0].position for node in all_nodes])
        node_ids = [node[0].sequential_id for node in all_nodes]

        cell_ids = [cell_id for _, cell_id in all_nodes]
        self.adhesions = [] if self.step %1000 ==0 else self.adhesions
        # Build KDTree based on node positions
        tree = KDTree(node_positions)

        # Dictionary to store neighboring nodes
        node_neighbors = {}
        for i, node_position in enumerate(node_positions):
            distances, indices = tree.query(node_position, k=len(node_positions))
            nearest_neighbor = None
            for dist, idx in zip(distances[1:], indices[1:]):
                if dist > 20:  # Threshold distance for node neighbors
                    break
                if cell_ids[idx] != cell_ids[i]:  # Skip if from the same cell
                    nearest_neighbor = node_ids[idx]
                    break
            if nearest_neighbor:
                node_neighbors[node_ids[i]] = nearest_neighbor
                # Add logging for neighbors
                logger.info(f"Node {node_ids[i]} has neighbor Node {nearest_neighbor}")
        neighbor_counts = {}
        for node_id, neighbor_id in node_neighbors.items():
            if neighbor_id in neighbor_counts:
                neighbor_counts[neighbor_id] += 1
            else:
                neighbor_counts[neighbor_id] = 1

        for neighbor, count in neighbor_counts.items():
            logger.info(f"Node {neighbor} is a neighbor to {count} other nodes")

        # Fix for the tuple issue
        nodes_dict = {node[0].sequential_id: node[0] for node in all_nodes}

        if not self.adhesions or True:
            for node_id, neighbor_id in node_neighbors.items():
                ad_fil = Filament(node1=nodes_dict[node_id], node2=nodes_dict[neighbor_id], cell_id=None,
                                  type="adhesion")
                if ad_fil is not None:
                    logger.info(f"Add filament adhesion: {ad_fil}")

                    self.adhesions.append(ad_fil)


        logger.info(f"node_neighbors: {node_neighbors}")
        logger.info(f"Updated adhesions: {len(self.adhesions)}")

        return

    def create_an_eptm_of_a_growing_cells(self):
        """
        Creates an epithelial tissue model with one growing cells.
        """
        cell1 = Cell()
        cell1.redefine_volume0()
        # translate out of the plane
        cell1.translate_cell([0, 0, 5])

        self.cells.append(cell1)

    def create_an_eptm_of_two_growing_cells(self):
        """
        Creates an epithelial tissue model with two growing cells.
        """
        cell1 = Cell()
        cell2 = Cell()

        translation_vector = [0, 40, 0]  # Translate +20 on the z-axis
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
        self.filaments = []
        for cell in self.cells:
            self.filaments.extend(cell.get_filaments())  # Call the get_filaments method of each cell

        return self.filaments

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
        self.state = self.EulerOneD(self.state, t, dt)
        # Update Normals for facets
        for cell in self.cells:
            for facet in cell.facets:
                facet._calculate_normal()

        # grow cell to target volume
        for cell in self.cells:
            cell.grow_to_target_volume()

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

    def derive_state(self, state, t, debug=False):
        """
        Derives the state of the system, given in the form (position, velocity)
        In dynamics, the derived system is (velocity, acceleration), with acc = F/m
        Hence, it boils down to calculating the forces applied to the nodes.
        """
        logger.superior_info("Derive state")

        state2 = state.copy() * 0.

        # Gather all nodes and filaments
        Nodes = self.nodes
        Filaments = self.get_all_filaments()
        # Update adhesion forces
        self.update_adhesion()
        Filaments.extend(self.adhesions)



        # Pre-calculate forces
        gravity = params["gravity"]
        viscosity = params["medium_viscosity"]

        # Initialize force arrays for debugging
        ground_contact_forces = []
        gravity_forces = []
        viscous_damping_forces = []
        elastic_forces = []

        # Reset forces and apply ground contact forces in a single loop
        for node in Nodes:
            node.reset_forces()
            gcf = node.ground_contact_force()
            node.forces += gcf
            ground_contact_forces.append(gcf)

        for node in Nodes:
            gf = node.gravity_force(gravity)
            vdf = node.viscous_damping_force(viscosity)
            node.forces += gf + vdf
            gravity_forces.append(gf)
            viscous_damping_forces.append(vdf)

        # Calculate filament forces and apply them also increment filament age
        for filament in Filaments:

            filament.increment_age()
            force = filament.elastic_force()
            filament.node1.forces += force
            filament.node2.forces -= force
            elastic_forces.append(force)

        # Calculate and apply pressure forces for each cell
        for cell in self.cells:
            cell.calculate_pressure()
            cell.apply_pressure_to_nodes()

        # Add pressure forces to nodes
        for node in Nodes:
            node.add_force(node.pressure_forces)



        # Detailed force logging for debugging
        if debug:
            for i, node in enumerate(Nodes):
                logger.debug(f"Node {i} Ground Contact Force: {ground_contact_forces[i]}")
                logger.debug(f"Node {i} Gravity Force: {gravity_forces[i]}")
                logger.debug(f"Node {i} Viscous Damping Force: {viscous_damping_forces[i]}")
                logger.debug(f"Node {i} Total Forces: {node.forces}")

            for i, filament in enumerate(Filaments):
                logger.debug(f"Filament {i} Elastic Force: {elastic_forces[i]}")

        # Calculate the derivative state
        mass_nodes = np.array([node.properties['mass'] for node in Nodes]).reshape(-1, 1)
        forces_nodes = np.array([node.forces for node in Nodes])

        # Update the state
        state2[1] = forces_nodes / mass_nodes
        state2[0] = state[1].copy()

        return state2



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

    def model_eptm_behaviour(self, t):
        """
        Args :
            t (float): simulation time step

        Run all the behaviour functionalities of the epithelium.
        """
        logger.superior_info(f"###Time {t} ###")
        for cell in self.cells:
            print(f"Cell behavior")
            if cell.division_counter <= 10000:
                cell.cell_behaviour()

    # functions for debbug
    def visualize_filament_forces(self, Nodes, Filaments, plot_resultant_forces=False):
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
                      length=0.1, color=color, linestyle='--')

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

    def fast_export_facets(self, output_dir='output/', filename='epithelium_facets', step=0):
        # Remove the output directory if it exists
        if os.path.exists(output_dir) and step == 1:
            shutil.rmtree(output_dir)

        # Recreate the output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Process and export each cell as a separate file
        for cell_idx, cell in enumerate(self.cells):
            points = []
            faces = []
            normals = []

            # Collect all points, normals, and faces in the cell
            for facet in cell.facets:
                facet_points = [node.position for node in facet.get_unique_nodes()]
                normal = facet.normal

                # Ensure triangle cell format [3, pt1, pt2, pt3]
                faces.append([len(facet_points)] + list(range(len(points), len(points) + len(facet_points))))
                points.extend(facet_points)

                # Repeat the normal for each face
                normals.append(normal)

            # Create the PyVista PolyData object
            poly_data = pv.PolyData(points, faces)

            # Assign normals to cell_data, since they correspond to facets (faces)
            poly_data.cell_data['Normals'] = normals

            # Save each cell to a VTK PolyData file (.vtp) for fast access in ParaView
            output_filepath = os.path.join(output_dir, f"{filename}_cell_{cell_idx}_{step}.vtp")
            poly_data.save(output_filepath, binary=True)
            print(f"Cell {cell_idx} facets saved as: {output_filepath}")

        # Exporting adhesions separately
        adhesion_points = []
        adhesion_lines = []

        for adhesion in self.adhesions:
            node1_pos = adhesion.node1.position
            node2_pos = adhesion.node2.position

            # Add positions of the adhesion filament nodes
            adhesion_points.extend([node1_pos, node2_pos])

            # Create the line between the two nodes
            line = [2, len(adhesion_points) - 2, len(adhesion_points) - 1]
            adhesion_lines.append(line)

        # Create PyVista PolyData object for adhesions
        if adhesion_points and adhesion_lines:
            adhesion_poly_data = pv.PolyData(adhesion_points, adhesion_lines)
        else:
            # If no adhesion data, create an empty PolyData object
            adhesion_poly_data = pv.PolyData()
        # Save adhesions to a separate file
        adhesion_output_filepath = os.path.join(output_dir, f"{filename}_adhesions_{step}.vtp")
        adhesion_poly_data.save(adhesion_output_filepath, binary=True)
        print(f"Adhesions saved as: {adhesion_output_filepath}")


        print(f"All facets and adhesions exported for step {step}.")

# ------------------------------------------------------------------------------