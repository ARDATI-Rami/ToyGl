#!/usr/bin/env python
# -*- coding:Utf-8 -*-

# pyToyGL
# J. Averseng
# LMGC - Univ. Montpellier - France

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import time, random, math

# variables globales
window = 0
screenSize = [640, 480]

previousTime = time.time()

frameCounter = 0
frameCounterTimer = time.time()

delta_time_ccc = 1

distance = 10
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
params = {"gravity": -200, "medium_viscosity": 8}


# Useful functions
def squared_norm(vec):
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]


def norm(vec):
    return np.sqrt(squared_norm(vec))


def alpha_beta_gamma_vn_tri_pt(pta, ptb, ptc, pt0):
    """Return the alpha and beta factors such as AP = alpha*AB + beta*AC
    in the triangle ABC. And gamma the perpendicular distance

    Args:
        pta (_type_): _description_
        ptb (_type_): _description_
        ptc (_type_): _description_
        pt0 (_type_): _description_

    Returns:

    """
    vab = np.array(ptb) - np.array(pta)
    vac = np.array(ptc) - np.array(pta)
    vap = np.array(pt0) - np.array(pta)
    vn = np.cross(vab, vac)
    vn *= 1. / norm(vn)

    gamma = vap.dot(vn)
    vam = vap - gamma * vn

    x_ac, y_ac = vac[0], vac[1]
    x_ab, y_ab = vab[0], vab[1]
    x_am, y_am = vam[0], vam[1]

    alpha = (x_ac * y_am - x_am * y_ac) / (-x_ab * y_ac + x_ac * y_ab)
    beta = (x_ab * y_am - x_am * y_ab) / (x_ab * y_ac - x_ac * y_ab)

    return [alpha, beta, gamma, vn]


# cell class
class Cell:
    """
    A class to describe a 3D cell
    """
    liste_tri3 = []  # list of 3 nodes ids describing a facet, facing outside
    volume = 1.  # calculé à chaque pas de temps
    volume0 = 20.
    dp_dv = 10.  # rigidité
    pression = 0.

    def __init__(self):
        """
        Init cell with a single tetraedra
        """
        liste_tri3 = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]


# system class
class Systeme:
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
    element_properties_list = []
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

        # setting the families of nodes and elements
        for i in range(10):
            self.node_properties_list.append({"radius": 1., "mass": 1., "stiffness": 100, "color": (1., 0., 0.)})
            self.element_properties_list.append({"radius": .5, "linear_mass": 1., "rig_EA": 100, "color": (1., 0., 0.)})

    def cell_divide_edge(self, cell_id, edge_id):
        """
        Division of the edge edge_id belonging to the cell cell_id

        Args:
            cell_id (_type_): concerned cell
            edge_id (_type_): concerned edge
        """
        # identification of concerned facets
        facets = []
        for id_tri, tri in enumerate(self.cell_list[cell_id].liste_tri3):
            for ij in [[0, 1], [1, 2], [2, 0]]:
                id_eli = self.element_index_ij(tri[ij[0]], tri[ij[1]])
                if id_eli == edge_id:
                    facets.append(id_tri)
                    # print(f"concerned facet {id_tri} ")

        tri1 = min(facets)
        tri2 = max(facets)
        # print(f"facets min/max = {tri1} et {tri2}")

        # nouveau noeud
        id1 = self.element_list[edge_id][0]
        id2 = self.element_list[edge_id][1]
        pt1 = self.node_positions[id1]
        pt2 = self.node_positions[id2]
        pt_nouv_noeud = 0.5 * (np.array(pt1) + np.array(pt2))
        id_nouv_noeud = self.add_node_family(pt_nouv_noeud, 0)

        # tri1 --> tri11 et tri12
        facet1 = self.cell_list[cell_id].liste_tri3[tri1]
        while ((facet1[0] == id1) or (facet1[0] == id2)):
            facet1 = np.roll(facet1, 1)
        f11 = [facet1[0], facet1[1], id_nouv_noeud]
        f12 = [facet1[0], id_nouv_noeud, facet1[2]]

        # tri2 --> tri21 et tri22
        facet2 = self.cell_list[cell_id].liste_tri3[tri2]
        while ((facet2[0] == id1) or (facet2[0] == id2)):
            facet2 = np.roll(facet2, 1)
        f21 = [facet2[0], facet2[1], id_nouv_noeud]
        f22 = [facet2[0], id_nouv_noeud, facet2[2]]

        # nouveaux elements
        id_new_1 = self.add_element_ij_family(f11[1], f11[2], 0)
        id_new_2 = self.add_element_ij_family(f12[1], f12[2], 0)
        id_new_11 = self.add_element_ij_family(f12[0], f12[1], 0)
        id_new_22 = self.add_element_ij_family(f22[0], f22[1], 0)

        # ajustement lfree
        lfree_edge = self.element_list[edge_id][3]
        self.element_list[id_new_1][3] = lfree_edge / 2.
        self.element_list[id_new_2][3] = lfree_edge / 2.
        self.element_list[id_new_11][3] *= 0.9
        self.element_list[id_new_22][3] *= 0.9

        # nouvelles facets
        self.cell_list[cell_id].liste_tri3.append(f11)
        self.cell_list[cell_id].liste_tri3.append(f12)
        self.cell_list[cell_id].liste_tri3.append(f21)
        self.cell_list[cell_id].liste_tri3.append(f22)

        # supprime facets 2 puis 1
        self.cell_list[cell_id].liste_tri3.pop(tri2)
        self.cell_list[cell_id].liste_tri3.pop(tri1)

        # supprime element
        self.delete_element(edge_id)

    def cell_uniformize_lfree(self, cell_id, l0):
        """
        Uniformizes the lfree  of the elements of the cell

        Args:
            cell_id (_type_): id of the cell
        """
        cell = self.cell_list[cell_id]
        for tri in cell.liste_tri3:
            for ij in [[0, 1], [1, 2], [2, 0]]:
                elem_id = self.element_index_ij(tri[ij[0]], tri[ij[1]])
                self.element_list[elem_id][3] = l0

    def cell_calculate_volume(self, cell_id):
        """
        Calculates the volume of the cell cell_id

        Args:
            cell_id (_type_): id of the cell
        """
        vol = 0
        cell = self.cell_list[cell_id]
        id_p0 = cell.liste_tri3[0][0]
        pt0 = np.array(self.node_positions[id_p0])
        for tri in cell.liste_tri3:
            pt1 = np.array(self.node_positions[tri[0]])
            pt2 = np.array(self.node_positions[tri[1]])
            pt3 = np.array(self.node_positions[tri[2]])
            v12 = pt2 - pt1
            v13 = pt3 - pt1
            v01 = pt1 - pt0
            vn = np.cross(v12, v13)
            norme_vn = norm(vn)
            base = 0.5 * norme_vn
            hauteur = np.dot(vn, v01) / norme_vn
            vol += base * hauteur / 3
        cell.volume = vol

    def add_base_cell(self, pt0, l0):
        """
        Adds a base cell = a simple tetrahedron
        made of side l0 from point pt0

        Args:
            pt0 (Vec3): insertion point
            l0 (float): side
        """
        pt1 = pt0 + np.array([l0, 0, 0])
        pt2 = pt0 + np.array([0, l0, 0])
        pt3 = pt0 + np.array([0, 0, l0])
        fam_no = 0
        fam_el = 0
        id0 = self.add_node_family(pt0, fam_no)
        id1 = self.add_node_family(pt1, fam_no)
        id2 = self.add_node_family(pt2, fam_no)
        id3 = self.add_node_family(pt3, fam_no)
        ide0 = self.add_element_ij_family(id0, id1, fam_el)
        ide1 = self.add_element_ij_family(id1, id2, fam_el)
        ide2 = self.add_element_ij_family(id2, id0, fam_el)
        ide3 = self.add_element_ij_family(id0, id3, fam_el)
        ide4 = self.add_element_ij_family(id1, id3, fam_el)
        ide5 = self.add_element_ij_family(id2, id3, fam_el)

        cell0 = Cell()
        cell0.liste_tri3 = [[id0, id2, id1], [id0, id1, id3], [id1, id2, id3], [id0, id3, id2]]
        self.cell_list.append(cell0)
        return len(self.cell_list) - 1

    def delete_element(self, id_el):
        """
        Deletes the specified element

        Args:
            elem_id (_type_): index of the element
        """
        # traitement de element_list
        self.element_list.pop(id_el)

        # traitement de liste_elements_bloques
        id_pop = -1
        for i, idsel in enumerate(self.selected_elements_list):
            if idsel == id_el:
                id_pop = i
            if idsel > id_el:
                self.selected_elements_list[i] -= 1
        if id_pop >= 0:
            self.selected_elements_list.pop(id_pop)

    def delete_node(self, node_id):
        """
        Deletes the node node_id

        Args:
            node_id (_type_): index of the node
        """
        # traitement pos_noeud, vit_noeud
        self.node_positions.pop(node_id)
        self.node_velocities.pop(node_id)

        # traitement self.list_blocked_nodes
        id_pop = -1
        for i, idi in enumerate(self.list_blocked_nodes):
            if idi == node_id:
                id_pop = i
            if idi > node_id:
                self.list_blocked_nodes[i] -= 1
        if id_pop >= 0:
            self.list_blocked_nodes.pop(id_pop)

        # traitement self.selected_nodes_list
        id_pop = -1
        for i, idi in enumerate(self.selected_nodes_list):
            if idi == node_id:
                id_pop = i
            if idi > node_id:
                self.selected_nodes_list[i] -= 1
        if id_pop >= 0:
            self.selected_nodes_list.pop(id_pop)

        # traitement self.cell_list
        #      decalage indices seulement
        #      il conviendrait de verifier les elements aussi
        for cell in self.cell_list:
            for tri in cell.liste_tri3:
                for i in range(3):
                    if tri[i] > node_id:
                        tri[i] -= 1

    def add_node_family(self, vec3, fam):
        """
        Adds a node with coordinates vec3 of the given family
        """
        ind_retour = self.node_index(vec3)
        if ind_retour == -1:
            self.node_positions.append(vec3)
            self.node_families.append(fam)
            self.node_velocities.append([0., 0., 0.])
            ind_retour = len(self.node_positions) - 1
        else:
            # the node exists but we assign it the new family
            # print("node exists...")
            self.node_families[ind_retour] = fam
        return ind_retour

    def add_element_ij_family(self, i, j, fam):
        """
        Adds an element between node i and node j of family family
        """
        # print("add elem", i, j, fam)
        nb_nodes = len(self.node_positions)
        # print("nb_nodes", nb_nodes)
        if (i >= 0) and (i < nb_nodes) and (j >= 0) and (j < nb_nodes) and (i != j):
            # print("pour l'instant ca va")
            # on peut y aller
            ind_existe_deja = self.element_index_ij(i, j)
            existe_deja = False
            if ind_existe_deja >= 0 and self.element_list[ind_existe_deja][2] == fam:
                # print("en plus il est de la meme famille")
                existe_deja = True
            if (not existe_deja) and (fam >= 0) and (fam < len(self.element_properties_list)):
                # on peut l'ajouter
                # calcul de la longueur
                vdir = np.array(self.node_positions[i]) - np.array(self.node_positions[j])
                self.element_list.append([i, j, fam, norm(vdir)])
                ind_existe_deja = len(self.element_list) - 1
            return ind_existe_deja

    def add_element_p1p2_fn1_fn2_felem(self, pt1, pt2, fn1, fn2, fel):
        """
        Adds an element between nodes pt1 and pt2
        Takes care of verifying their existence first
        """
        ind1 = self.add_node_family(pt1, fn1)
        ind2 = self.add_node_family(pt2, fn2)
        ind_retour = self.add_element_ij_family(ind1, ind2, fel)
        return ind_retour

    def lock_node(self, indice_noeud, bx, by, bz):
        """
        Locks the node according to bx, by, bz
        """
        if indice_noeud >= 0 and indice_noeud < len(self.node_positions):
            # indice_noeud compatible, on peut y aller
            node_index = -1
            for ind, no in enumerate(self.list_blocked_nodes):
                if no[0] == indice_noeud:
                    node_index = ind
            if node_index == -1:
                self.list_blocked_nodes.append([indice_noeud, bx, by, bz])
            else:
                self.list_blocked_nodes[node_index] = [indice_noeud, bx, by, bz]

    def element_index_ij(self, ind1, ind2):
        """
        Returns the index of a geometric element (ind1, ind2), if it exists
        """
        ind_retour = -1
        for ind0, el in enumerate(self.element_list):
            if ((el[0] == ind1 and el[1] == ind2) or (el[1] == ind1 and el[0] == ind2)):
                ind_retour = ind0
        # if ind_retour >= 0 :
        #     print("on a identifie un element")
        return ind_retour

    def node_index(self, vec3, tol=1.E-6):
        """
        Returns the index of the node at point vec3, within a given tolerance
        """
        index_retour = -1
        # lent
        for ind, no in enumerate(self.node_positions):
            if squared_norm(np.array(vec3) - np.array(no)) < tol * tol:
                index_retour = ind
        return index_retour

    def set_node_family_characteristics(self, fam, radius, mass, stiffness, color):
        """
        Sets the characteristics of the node family family
        """
        self.node_properties_list[fam]["radius"] = radius
        self.node_properties_list[fam]["mass"] = mass
        self.node_properties_list[fam]["stiffness"] = stiffness
        self.node_properties_list[fam]["color"] = color

    def set_element_family_characteristics(self, fam, radius, linear_mass, rig_EA, color):
        """
        Sets the characteristics of the element family family
        """
        self.element_properties_list[fam]["radius"] = radius
        self.element_properties_list[fam]["linear_mass"] = linear_mass
        self.element_properties_list[fam]["rig_EA"] = rig_EA
        self.element_properties_list[fam]["color"] = color

    def drawGL_cells(self):
        """
        Draws the cells in triangle shape
        """
        for cell in self.cell_list:
            for tri in cell.liste_tri3:
                pt1 = self.node_positions[tri[0]]
                pt2 = self.node_positions[tri[1]]
                pt3 = self.node_positions[tri[2]]
                glColor4f(0, 1, 0, 0.5)
                draw_triangle(pt1, pt2, pt3)

    def drawGL_nodes(self):
        """
        Draws the elements in the form of a sphere
        """
        for ind, pos in enumerate(self.node_positions):
            carac = self.node_properties_list[self.node_families[ind]]
            radius = carac["radius"]
            color = carac["color"]
            if ind in self.selected_nodes_list:
                glColor3f(1, 1, 0)
                radius *= 1.1
            else:
                glColor3f(color[0], color[1], color[2])
            if ind in self.contact_nodes_list:
                glColor3f(1, 0, 0)
                radius *= 1.5
            glLoadName(ind)
            draw_sphere(pos, radius)

    def drawGL_elements_lines(self):
        """
        Draws the elements in the form of a line
        """
        for el in self.element_list:
            no_deb = self.node_positions[el[0]]
            no_fin = self.node_positions[el[1]]
            glBegin(GL_LINES)
            glVertex3f(no_deb[0], no_deb[1], no_deb[2])
            glVertex3f(no_fin[0], no_fin[1], no_fin[2])
            glEnd()
            # draw_cylinder(no_deb,no_fin,1)

    def drawGL_elements_tubes(self):
        """
        Draws the elements in the form of a tube
        """
        for ind, el in enumerate(self.element_list):
            no_deb = self.node_positions[el[0]]
            no_fin = self.node_positions[el[1]]
            # rayon1 = self.node_properties_list[self.node_families[el[0]]]["radius"]
            # rayon2 = self.node_properties_list[self.node_families[el[1]]]["radius"]
            rayon_elem = self.element_properties_list[el[2]]["radius"]
            color = self.element_properties_list[el[2]]["color"]
            if ind in self.selected_elements_list:
                glColor3f(1, 1, 0)
                rayon_elem *= 1.1
            else:
                glColor3f(color[0], color[1], color[2])
            glLoadName(ind)
            draw_cylinder(no_deb, no_fin, rayon_elem, rayon_elem)

    def drawGL_node_locks(self):
        """
        Draws a black cone along each locked axis for all locked nodes
        """
        glColor3f(0., 0., 0.)
        glDisable(GL_CULL_FACE)
        for nob in self.list_blocked_nodes:
            pt = self.node_positions[nob[0]]
            radius = self.node_properties_list[self.node_families[nob[0]]]["radius"]
            if nob[1] > 0:
                # dessine un cone selon x
                glPushMatrix()
                glTranslatef(pt[0] - 2 * radius, pt[1], pt[2])
                glRotatef(90., 0., 1., 0.)
                glutSolidCone(radius, radius, 10, 10)
                glPopMatrix()
            if nob[2] > 0:
                # dessine un cone selon y
                glPushMatrix()
                glTranslatef(pt[0], pt[1] - 2 * radius, pt[2])
                glRotatef(-90., 1., 0., 0.)
                glutSolidCone(radius, radius, 10, 10)
                glPopMatrix()

            if nob[3] > 0:
                # dessine un cone selon z
                glPushMatrix()
                glTranslatef(pt[0], pt[1], pt[2] - 2 * radius)
                # glRotatef(90.,0.,.,1.)
                glutSolidCone(radius, radius, 10, 10)
                glPopMatrix()
        glDisable(GL_CULL_FACE)

    def dynamic_evolution(self, t, dt):
        # pos0 = np.array(self.node_positions).copy()
        etat_actuel = np.array([self.node_positions, self.node_velocities]).copy()
        etat_suivant = self.RKOneD(etat_actuel, t, dt)
        self.node_positions = list(etat_suivant.copy()[0])
        self.node_velocities = list(etat_suivant.copy()[1])

        if flags["dynamic_relaxation"]:
            self.kinetic_energy.append(np.square(self.node_velocities).sum())
            # print(self.node_velocities)
            if len(self.kinetic_energy) > 3:
                # print(self.kinetic_energy)
                self.kinetic_energy.pop(0)
                if (self.kinetic_energy[1] > self.kinetic_energy[0]) and (
                        self.kinetic_energy[1] > self.kinetic_energy[0]):
                    for vit in self.node_velocities:
                        vit *= 0.

    # integrateurs temporel
    def RKOneD(self, x, t, dt):
        """
        RungeKutta ordre 4. necessite des etats de type np.array
        """
        k1 = dt * self.derive_state(x, t)
        k2 = dt * self.derive_state(x + k1 / 2.0, t)
        k3 = dt * self.derive_state(x + k2 / 2.0, t)
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
        state2 = state.copy() * 0.
        nodes_forces = state2[0].copy()

        nb_nodes = len(state[0])

        # calculation of forces at nodes
        for ind in range(nb_nodes):
            # ground contact
            car_no = self.node_properties_list[self.node_families[ind]]
            dz = state[0][ind][2] - car_no['radius']
            if dz < 0.:
                nodes_forces[ind] -= np.array([0., 0., car_no['stiffness'] * dz])

            # gravity
            nodes_forces[ind] += np.array([0., 0., params["gravity"] * car_no['mass']])

            # viscous damping
            nodes_forces[ind] -= np.array(state[1][ind]) * params["medium_viscosity"]

        # forces due to elements
        self.fixed_element_list = [[0, 0, 0]] * len(self.element_list)
        for ind, el in enumerate(self.element_list):
            car_el = self.element_properties_list[el[2]]
            ind0 = el[0]
            ind1 = el[1]
            l0 = el[3]
            v_elem = np.array(state[0][ind1]) - np.array(state[0][ind0])
            l_elem = norm(v_elem)
            v_force = v_elem * car_el['rig_EA'] * (l_elem - l0) / (l0 * l_elem)
            self.fixed_element_list[ind] = v_force
            nodes_forces[ind0] += v_force
            nodes_forces[ind1] -= v_force

        # pressure forces
        for cell_id, cell in enumerate(self.cell_list):
            # calcul volume
            self.cell_calculate_volume(cell_id)
            # print(f"Le volume de la cellule {cell_id} est {cell.volume}")

            # calculate pressure
            v0 = cell.volume0
            cell.pression = (cell.volume0 - cell.volume) * cell.dp_dv
            # print(f"           et sa pression est {cell.pression}")

            # application to nodes
            for tri in cell.liste_tri3:
                pt0 = np.array(self.node_positions[tri[0]])
                pt1 = np.array(self.node_positions[tri[1]])
                pt2 = np.array(self.node_positions[tri[2]])
                v01 = pt1 - pt0
                v02 = pt2 - pt0
                vn = np.cross(v01, v02)
                # vn_norme = norm(vn)
                v_force = cell.pression * vn / 2
                nodes_forces[tri[0]] += v_force / 3
                nodes_forces[tri[1]] += v_force / 3
                nodes_forces[tri[2]] += v_force / 3

        # node-to-node contact
        if flags["node-node_contact"]:
            for i in range(nb_nodes - 1):
                fam_i = self.node_families[i]
                car_i = self.node_properties_list[fam_i]
                r_i = car_i['radius']
                for j in range(i + 1, nb_nodes):
                    fam_j = self.node_families[j]
                    car_j = self.node_properties_list[fam_j]
                    r_j = car_j['radius']
                    dmin = r_i + r_j
                    vdir = state[0][j] - state[0][i]
                    if not (fam_i == 1 and fam_j == 1):
                        if abs(vdir[0]) < dmin:
                            if abs(vdir[1]) < dmin:
                                if abs(vdir[2]) < dmin:
                                    distsq = sum(vdir * vdir)
                                    gap = distsq - (r_i + r_j) * (r_i + r_j)
                                    if gap < 0:
                                        # contact
                                        k_i = car_i['stiffness']
                                        k_j = car_j['stiffness']
                                        dist = np.sqrt(distsq)
                                        force = (1 / (1 / k_i + 1 / k_j)) * (dist - np.sqrt((r_i + r_j) * (r_i + r_j)))
                                        nodes_forces[i] += force * vdir / dist
                                        nodes_forces[j] -= force * vdir / dist

        # contact cell-cell
        tcc0 = time.time()
        if (flags["cell-cell_contact"]) and t > 0.5:
            self.contact_nodes_list = []
            r_no = self.node_properties_list[0]["radius"]
            k_no = self.node_properties_list[0]["stiffness"]
            nb_cells = len(self.cell_list)
            for i in range(nb_cells - 1):
                cell_i = self.cell_list[i]
                for j in range(i + 1, nb_cells):
                    cell_j = self.cell_list[j]
                    for tri3_i in cell_i.liste_tri3:
                        ptsi = [self.node_positions[i] for i in tri3_i]
                        cdg_i = 0.333333 * sum(ptsi)
                        for tri3_j in cell_j.liste_tri3:
                            ptsj = [self.node_positions[i] for i in tri3_j]
                            cdg_j = 0.333333 * sum(ptsj)
                            dij = cdg_j - cdg_i
                            lij2 = squared_norm(dij)
                            if lij2 < 1:
                                # contact possible
                                # # traitement noeud i
                                # for ii,pti in enumerate(ptsi):
                                #     vgipi = pti - cdg_i
                                #     fpi = -kcc * dij * max(np.dot(vgipi,dij),0) / lij2
                                #     nodes_forces[tri3_i[ii]] += fpi
                                # for jj,ptj in enumerate(ptsj):
                                #     vgjpj = ptj - cdg_j
                                #     fpj = -kcc * dij * min(np.dot(vgjpj,dij),0) / lij2
                                #     nodes_forces[tri3_j[jj]] += fpj

                                # nodes of triangle i with triangle j
                                for ii, pti in enumerate(ptsi):
                                    alpha_i, beta_i, gamma_i, vn_j = alpha_beta_gamma_vn_tri_pt(ptsj[0], ptsj[1],
                                                                                                ptsj[2], pti)
                                    if gamma_i > -1.5:
                                        if gamma_i < r_no * 1.5:
                                            if alpha_i >= 0:
                                                if beta_i >= 0:
                                                    if alpha_i + beta_i <= 1:
                                                        # print(f"possible contact with {alpha_i}{beta_i}{gamma_i}")
                                                        # self.contact_nodes_list.append((tri3_i[ii],tri3_j))

                                                        # contact management - calculation of contact force and resultant forces
                                                        R = (gamma_i - r_no) * k_no
                                                        RA = R * (1 - alpha_i - beta_i)
                                                        RB = R * alpha_i
                                                        RC = R * beta_i

                                                        # application of forces
                                                        nodes_forces[tri3_j[0]] += vn_j * RA
                                                        nodes_forces[tri3_j[1]] += vn_j * RB
                                                        nodes_forces[tri3_j[2]] += vn_j * RC
                                                        nodes_forces[tri3_i[ii]] -= vn_j * R
                                # nodes of triangle j with triangle i
                                for jj, ptj in enumerate(ptsj):
                                    alpha_j, beta_j, gamma_j, vn_i = alpha_beta_gamma_vn_tri_pt(ptsi[0], ptsi[1],
                                                                                                ptsi[2], ptj)
                                    if gamma_j > -1.5:
                                        if gamma_j < r_no * 1.5:
                                            if alpha_j >= 0:
                                                if beta_j >= 0:
                                                    if alpha_j + beta_j <= 1:
                                                        # print(f"possible contact with {alpha_j}{beta_j}{gamma_j}")
                                                        # self.contact_nodes_list.append((tri3_j[jj],tri3_i))
                                                        # contact management - calculation of contact force and resultant forces
                                                        R = (gamma_j - r_no) * k_no
                                                        RA = R * (1 - alpha_j - beta_j)
                                                        RB = R * alpha_j
                                                        RC = R * beta_j

                                                        # application of forces
                                                        nodes_forces[tri3_i[0]] += vn_i * RA
                                                        nodes_forces[tri3_i[1]] += vn_i * RB
                                                        nodes_forces[tri3_i[2]] += vn_i * RC
                                                        nodes_forces[tri3_j[jj]] -= vn_i * R
            # if len(self.contact_nodes_list) > 0:
            #    print(f"List nodes contact = {self.contact_nodes_list}")
        tcc1 = time.time()
        global delta_time_ccc
        delta_time_ccc = tcc1 - tcc0

        ########################################################################
        # calculation of the derivative state
        for ind in range(len(state[0])):
            masse_no = self.node_properties_list[self.node_families[ind]]['mass']
            state2[1][ind] = nodes_forces[ind] / masse_no
            state2[0][ind] = state[1][ind].copy()

        # taking into account the blocked nodes
        for nob in self.list_blocked_nodes:
            for j in range(3):
                if nob[j + 1] != 0:
                    state2[0][nob[0]][j] = 0

        return state2

    def display_stats(self):
        """
        Displays statistics on the current state
        """
        # position of the selected node
        if len(self.selected_nodes_list) == 1:
            no_sel = self.selected_nodes_list[0]
            pt_no_sel = self.node_positions[no_sel]
            print("Node ", no_sel, pt_no_sel)

        # L, lfree and force in the selected element
        if len(self.selected_elements_list) == 1:
            el_sel = self.selected_elements_list[0]
            el = self.element_list[el_sel]
            pt1 = self.node_positions[el[0]]
            pt2 = self.node_positions[el[1]]
            fam = el[2]
            lfree = el[3]
            car = self.element_properties_list[fam]
            l = norm(np.array(pt2) - np.array(pt1))
            n = car["rig_EA"] * (l - lfree) / lfree
            print("Element ", el_sel, " - L = ", l, " - lfree = ", lfree, " - T = ", n)

        # pressions des cellules
        for i_cell, celli in enumerate(self.cell_list):
            print(f" - cell {i_cell} - pressure={celli.pression} - volume={celli.volume} - volume0={celli.volume0}")


# ------------------------------------------------------------------------------


def setCamera():
    # print " camera :"
    # print "   dist : ", distance
    # print "   elev : ", elev
    # print "   azim : ", azim
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0., 0., -distance)
    glRotatef(elev, 1., 0., 0.)
    glRotatef(azim, 0., 0., 1.)
    glTranslatef(origine[0], origine[1], origine[2])


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


def draw_reference_frame():
    glBegin(GL_LINES)
    glColor3f(1., 0., 0.)
    glVertex3f(0., 0., 0.)
    glVertex3f(1., 0., 0.)
    glColor3f(0., 1., 0.)
    glVertex3f(0., 0., 0.)
    glVertex3f(0., 1., 0.)
    glColor3f(0., 0., 1.)
    glVertex3f(0., 0., 0.)
    glVertex3f(0., 0., 1.)
    glEnd()


def draw_triangle(pt1, pt2, pt3):
    glBegin(GL_TRIANGLES)
    glVertex3f(pt1[0], pt1[1], pt1[2])
    glVertex3f(pt2[0], pt2[1], pt2[2])
    glVertex3f(pt3[0], pt3[1], pt3[2])
    glEnd()

    # et la normale en plus
    v0 = (np.array(pt1) + np.array(pt2) + np.array(pt3)) / 3
    vn = np.cross(np.array(pt2 - pt1), np.array(pt3 - pt1))
    vn = vn / norm(vn)
    v1 = v0 + vn
    glBegin(GL_LINES)
    glVertex3f(v0[0], v0[1], v0[2])
    glVertex3f(v1[0], v1[1], v1[2])
    glEnd()


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
    # premaration timing
    global previousTime, frameCounter, frameCounterTimer, envlist
    currentTime = time.time()
    deltaTime = (currentTime - previousTime)

    # effacement + placement camera
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    setCamera()

    #########################################
    # dessin proprement dit
    mon_sys.drawGL_nodes()
    mon_sys.drawGL_elements_tubes()
    mon_sys.drawGL_node_locks()
    draw_reference_frame()
    draw_plane()

    mon_sys.drawGL_cells()

    #########################################

    # basculement buffer : affichage
    glutSwapBuffers()

    # timing
    previousTime = currentTime
    frameCounter += 1
    if currentTime - frameCounterTimer > 1:
        # affichage toutes les secondes
        print("FPS:", frameCounter)
        print(f"t = {t} - delta_ccc = {delta_time_ccc}")
        frameCounter = 0
        frameCounterTimer = currentTime
        mon_sys.display_stats()


def resize(width, height):
    """
    Resizing the window
    """
    if (height == 0): height = 1
    screenSize = [width, height]
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(80.0, float(width) / float(height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)


def key(*args):
    """
    Keyboard management
    """
    print(f"function Key called with arguments {args}")

    if args[0] == b'w':
        params['gravity'] += 5
        print(f"{params}")
    elif args[0] == b'W':
        params['gravity'] -= 5
        print(f"{params}")
    elif args[0] == b'r':
        print("relaxation dynamiqe")
        if flags["dynamic_relaxation"]:
            flags["dynamic_relaxation"] = False
        else:
            flags["dynamic_relaxation"] = True
        print(f"{flags}")
    elif args[0] == b'c':
        print("cell-cell_contact")
        if flags["cell-cell_contact"]:
            flags["cell-cell_contact"] = False
        else:
            flags["cell-cell_contact"] = True
        print(f"{flags}")
    elif args[0] == b'v':
        params["medium_viscosity"] += 1
        print(f"{params}")
    elif args[0] == b'V':
        params["medium_viscosity"] -= 1
        print(f"{params}")
    elif args[0] == b'p':
        mon_sys.cell_list[0].volume0 += 1
        print(f"Volume0 = {mon_sys.cell_list[0].volume0}")
    elif args[0] == b'P':
        mon_sys.cell_list[0].volume0 -= 1
        print(f"Volume0 = {mon_sys.cell_list[0].volume0}")
    elif args[0] == b'\033':  # escape
        glutDestroyWindow(window)
        # sys.exit()


def mouse(*args):
    """
    Mouse management
    """

    global mouse_orig, rotate, zoom, translation

    button = args[0]
    state = args[1]
    x = args[2]
    y = args[3]

    if button == GLUT_LEFT_BUTTON:
        # rotation
        if (state == GLUT_DOWN):
            if glutGetModifiers() == GLUT_ACTIVE_CTRL:
                mouse_orig[2] = y
                zoom = True
            else:
                mouse_orig[0] = x
                mouse_orig[1] = y
                selection(mon_sys, x, y, 0, glutGetModifiers())
                rotate = True
        else:
            rotate = False
            zoom = False
            translation = False
    elif button == GLUT_RIGHT_BUTTON:
        if glutGetModifiers() == GLUT_ACTIVE_CTRL:
            # deplacement de l'origine
            translation = True
            mouse_orig[0] = x
            mouse_orig[1] = y
        else:
            # zoom
            if (state == GLUT_DOWN):
                mouse_orig[2] = y
                zoom = True
            else:
                zoom = False


def selection(systeme, x, y, type_sel, modificateur):
    """
    manages clicking on elements or nodes
    """
    no_selected, no_dist = selectionne(x, y, systeme.drawGL_nodes)
    el_selected, el_dist = selectionne(x, y, systeme.drawGL_elements_tubes)
    # print "no_selected =", no_selected
    # print "el_selected =", el_selected

    if (no_selected >= 0 or el_selected >= 0):
        if (no_dist < el_dist):
            # le noeud est le plus proche
            systeme.selected_elements_list = []
            if modificateur == GLUT_ACTIVE_SHIFT:
                # on fait une selection multiple
                if systeme.selected_nodes_list.count(no_selected) == 0:
                    systeme.selected_nodes_list.append(no_selected)
                else:
                    systeme.selected_nodes_list.remove(no_selected)
            else:
                systeme.selected_nodes_list = [no_selected]
        else:
            # l'element est le plus proche
            systeme.selected_nodes_list = []
            if modificateur == GLUT_ACTIVE_SHIFT:
                # on fait une selection multiple
                if systeme.selected_elements_list.count(el_selected) == 0:
                    systeme.selected_elements_list.append(el_selected)
                else:
                    systeme.selected_elements_list.remove(el_selected)
            else:
                systeme.selected_elements_list = [el_selected]
    else:
        print("rien de selectionne...")


def selectionne(x, y, dessine):
    """
    returns the number of the system node under the point x,y
    """
    buffsize = 512
    viewport = glGetIntegerv(GL_VIEWPORT)
    glSelectBuffer(buffsize)
    glRenderMode(GL_SELECT)
    glInitNames()
    glPushName(-1)

    # reglage de la matrice de projection restrictive
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    #  create 3x3 pixel picking region near cursor location
    gluPickMatrix(x, viewport[3] - y, 5.0, 5.0, viewport)

    # reglage perspective et camera
    gluPerspective(80., (1. * viewport[2] - 1. * viewport[0]) / (1. * viewport[3] - 1. * viewport[1]), .1, 100.)
    setCamera()

    # dessin pour selection
    dessine()

    # recupere les infos de selection
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    hits = glRenderMode(GL_RENDER)

    # traitement
    selectionne = -1
    min_depth = 10000.
    # print "--------- pick3d ----------"
    # print " - nhits =", len(hits )
    if len(hits) == 0:
        print("on a rien touché")
    else:
        min_depth = hits[0][0]
        selectionne = hits[0][2][0]
        for record in hits:
            minDepth, maxDepth, names = record
            if minDepth < min_depth:
                min_depth = minDepth
                selectionne = names[0]
                # print "selection brute = ", selectionne
    return selectionne, min_depth


def motion(*args):
    """
    Mouse movement management
    """
    global mouse_orig, rotate, zoom, transation, distance, elev, azim

    x = args[0]
    y = args[1]

    if rotate:
        azim = (azim + .2 * (x - mouse_orig[0]))
        elev = (elev + .2 * (y - mouse_orig[1]))
        azim = azim % 360.
        elev = elev % 360.
        mouse_orig[0] = x
        mouse_orig[1] = y
        glutPostRedisplay()

    if zoom:
        distance = distance + 0.01 * (y - mouse_orig[2]);
        mouse_orig[2] = y
        glutPostRedisplay()

    if translation:
        v = [sin(azim * pi / 180), cos(azim * pi / 180)]
        vp = [v[1], -v[0]]
        dx = (x - mouse_orig[0])
        dy = (y - mouse_orig[1])
        origine[0] += 0.1 * (dx * vp[0] - dy * v[0])
        origine[1] += 0.1 * (dx * vp[1] - dy * v[1])
        mouse_orig[0] = x
        mouse_orig[1] = y


def animation():
    """
    system animation
    """
    global t, dt, t_div_elem
    # boucle pour faire plusieurs calculs entre deux affichages
    for i in range(1):
        t += dt  # avancement du temps
        # mon_sys.node_positions[10][0] += 0.01
        mon_sys.dynamic_evolution(t, dt)  # evoluton dynamique du systeme
        # on peu rajouter ici une autre fonction d'evolution :
        #   inteligence artificielle du systeme
        comportement_cellule(mon_sys, t)
    display()


def comportement_cellule(syst, temps):
    """
    Management of cell behavior
    """
    global t, dt, t_div_elem

    # croissance du volume0 des cellules
    for cell in syst.cell_list:
        # print(f"le volume de la cellule en cours est {cell.volume}")
        if cell.pression < 1000 and cell.volume0 < 200:  # garde fou
            cell.volume0 += 50 * dt

    # croissance des lfree des elements
    elem_id_division = -1
    id_cell_division = -1
    llib_elem_division = 0
    l_elem_division = 0
    for cell_id, cell in enumerate(syst.cell_list):
        for tri in cell.liste_tri3:
            for ij in [[0, 1], [1, 2], [2, 0]]:
                node_i = tri[ij[0]]
                node_j = tri[ij[1]]
                elem_id = syst.element_index_ij(node_i, node_j)
                force = norm(syst.fixed_element_list[elem_id])
                llib_i = syst.element_list[elem_id][3]
                l_i = norm(np.array(syst.node_positions[node_i]) - np.array(syst.node_positions[node_j]))
                # print(f"Element {elem_id} - force = {force}")
                if force > 200:
                    # allongement
                    if syst.element_list[elem_id][3] < 1.2:  # garde fou
                        syst.element_list[elem_id][3] += 0.005
                if force < 100:
                    # raccourcissement
                    if syst.element_list[elem_id][3] < 0.5:  # garde fou
                        syst.element_list[elem_id][3] -= 0.005
                # division
                if l_i > 1.5:
                    if cell.volume < 200:
                        # if force > 500:
                        # uniquement si le plus long
                        # llib_i = syst.element_list[elem_id][3]
                        # if llib_i > llib_elem_division:
                        if l_i > l_elem_division:
                            elem_id_division = elem_id
                            id_cell_division = cell_id
                            llib_elem_division = llib_i
                            l_elem_division = l_i

    if elem_id_division >= 0:
        if t > t_div_elem + 10 * dt:  # garde fou
            # division de l'élément elem_id_division
            # print(f"Division de l'élément {elem_id_division} dans la cellule {id_cell_division}")
            syst.cell_divide_edge(id_cell_division, elem_id_division)
            t_div_elem = t
            # print("on vient de diviser")


if __name__ == '__main__':
    print("Hello World!")

    ######################################
    ### programme principal
    ####################################
    # intialisations GLUT
    glutInit('')
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    window = glutCreateWindow('PyToyGL')
    resize(640, 480)
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

    # reponse aux evenements
    glutDisplayFunc(display)
    glutIdleFunc(animation)
    glutReshapeFunc(resize)
    # glutIgnoreKeyRepeat(1)
    glutKeyboardFunc(key)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)

    # on fait un systeme
    mon_sys = Systeme()

    # setting the characteristics of node families
    # setting the characteristics of element families
    # custom families

    mon_sys.set_node_family_characteristics(0, .2, .2, 1000., (0.2, 0.2, 0.2))
    mon_sys.set_node_family_characteristics(1, 0.5, .1, 1000., (1., 0., 0.))
    mon_sys.set_node_family_characteristics(2, 0.5, .1, 1000., (0., 1., 0.))

    mon_sys.set_element_family_characteristics(0, 0.1, .2, 800, (1., 0., 0.))
    mon_sys.set_element_family_characteristics(1, 0.1, .1, 1000, (0., 0., 1.))
    mon_sys.set_element_family_characteristics(2, 0.2, .1, 100, (1., 0., 1.))

    # systeme de base
    # pt1 = np.array([0,0,0])
    # pt2 = np.array([2,0,0])
    # pt3 = np.array([0,2,0])
    # pt4 = np.array([0,0,2])
    #####mon_sys.add_node_family(pt1,0)
    # mon_sys.add_element_p1p2_fn1_fn2_felem(pt1,pt2,0,0,0)
    # mon_sys.add_element_p1p2_fn1_fn2_felem(pt2,pt3,0,0,0)
    # mon_sys.add_element_p1p2_fn1_fn2_felem(pt3,pt1,0,0,0)
    # mon_sys.add_element_p1p2_fn1_fn2_felem(pt1,pt4,0,0,0)
    # mon_sys.add_element_p1p2_fn1_fn2_felem(pt2,pt4,0,0,0)
    # mon_sys.add_element_p1p2_fn1_fn2_felem(pt3,pt4,0,0,0)

    # systeme = cellule de base ---------------------------------------
    # id_cell1 = mon_sys.add_base_cell([-2,-1,2],1)
    # mon_sys.cell_uniformize_lfree(id_cell1,2)

    # id_cell2 = mon_sys.add_base_cell([2,-1,2],1)
    # mon_sys.cell_uniformize_lfree(id_cell2,2)

    id_cell3 = mon_sys.add_base_cell([0, 0, 2], 1)
    mon_sys.cell_uniformize_lfree(id_cell3, 2)

    id_cell4 = mon_sys.add_base_cell([3, 0, 20], 1)
    mon_sys.cell_uniformize_lfree(id_cell4, 2)

    print(len(mon_sys.node_positions), " nodes")
    print(len(mon_sys.element_list), " elements")

    # lancement de GLUT
    glutMainLoop()
