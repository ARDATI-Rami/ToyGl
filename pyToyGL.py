#!/usr/bin/env python
# -*- coding:Utf-8 -*-

# pyToyGL
# J. Averseng
# LMGC - Univ. Montpellier - France

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import time,random,math

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
dt = 0.002**1
t_div_elem = 0

flags = {"relaxation_dynamique":False,
         "contact-cell-cell":True,'contact_noeud_noeud' : False}
params = {"gravite":-200,"viscosite_milieu":8}

# fonctions utilitaire
def norme_au_carre(vec):
    return vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]

def norme(vec):
    return np.sqrt(norme_au_carre(vec))

def alpha_beta_gamma_vn_tri_pt(pta,ptb,ptc,pt0):
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
    vn = np.cross(vab,vac)
    vn *= 1./norme(vn)
    
    gamma = vap.dot(vn)
    vam = vap - gamma*vn
    
    x_ac,y_ac = vac[0],vac[1]
    x_ab,y_ab = vab[0],vab[1]
    x_am,y_am = vam[0],vam[1]
    
    alpha = (x_ac*y_am - x_am*y_ac)/(-x_ab*y_ac + x_ac*y_ab)
    beta = (x_ab*y_am - x_am*y_ab)/(x_ab*y_ac - x_ac*y_ab)
    
    return [alpha,beta,gamma,vn]


# cell class
class Cell:
    """
    Une classe pour decrire une cellule 3d
    """
    liste_tri3 = []    # list of 3 nodes ids describing a facet, facing outside
    volume = 1. # calculé à chaque pas de temps
    volume0 = 20.
    dp_dv = 10. # rigidité
    pression = 0.
    
    def __init__(self):
        """
        Init cell with a single tetraedra
        """
        liste_tri3 = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

# system class
class Systeme:
    """
    Definition d'un systeme masses-ressorts
    pos_noeuds : (x,y,z)
    vit_noeuds : (vx,vy,vz)
    fam_noeuds : indice_famille dans liste_car_noeuds
    elements : no_deb, no_fin, famille_el, llib
    car_noeuds : "rayon":r, "masse":m, "rigidite":k,"couleur":(r,v,b)
    car_elements : "rayon":r,"masse_lin":ml,"rigidite":EA,"couleur":(r,v,b)
    liste_noeuds_bloques : indice_no, bx,by,bz (0 = libre, 1 = bloque)
    """
    
    pos_noeuds = []
    vit_noeuds = []
    Ecin = []
    fam_noeuds = []
    liste_elements = []
    liste_f_elements = []
    liste_car_noeud = []
    liste_car_elements = []
    liste_noeuds_bloques = []
    liste_noeuds_selectionnes = []
    liste_elements_selectionnes = []
    liste_cells = []
    liste_noeuds_contact = []
    
    def __init__(self):
        """
        Initialisation : réglage des valeurs par défaut
        ici : systeme elementaire = 1 masse
        """
        # un noeud
        #self.ajoute_noeud_famille((0., 0., 1.), 0)
        
        # reglage des familles de noeuds et d'elements
        for i in range(10):
            self.liste_car_noeud.append({"rayon":1., "masse":1., "rigidite":100, "couleur":(1., 0., 0.)})
            self.liste_car_elements.append({"rayon":.5, "masse_lin":1., "rig_EA":100, "couleur":(1., 0., 0.)})
    
    def cell_division_arete(self, id_cell, id_arete):
        """
        Division de l'arete id_arete appartenant à la cellule id_cell

        Args:
            id_cell (_type_): cellule concernee
            id_arete (_type_): arete concernee
        """
        # identification facettes concernées
        facettes = []
        for id_tri, tri in enumerate(self.liste_cells[id_cell].liste_tri3):
            for ij in [[0,1],[1,2],[2,0]]:
                id_eli = self.index_element_ij(tri[ij[0]],tri[ij[1]])
                if id_eli == id_arete:
                    facettes.append(id_tri)
                    # print(f"facette {id_tri} concernée")

        tri1 = min(facettes)
        tri2 = max(facettes)
        #print(f"facettes min/max = {tri1} et {tri2}")
        
        # nouveau noeud
        id1 = self.liste_elements[id_arete][0]
        id2 = self.liste_elements[id_arete][1]
        pt1 = self.pos_noeuds[id1]
        pt2 = self.pos_noeuds[id2]
        pt_nouv_noeud = 0.5*(np.array(pt1) + np.array(pt2))
        id_nouv_noeud = self.ajoute_noeud_famille(pt_nouv_noeud,0)
        
        # tri1 --> tri11 et tri12
        facette1 = self.liste_cells[id_cell].liste_tri3[tri1]
        while((facette1[0] == id1) or (facette1[0] == id2)):
            facette1 = np.roll(facette1,1)        
        f11 = [facette1[0],facette1[1],id_nouv_noeud]
        f12 = [facette1[0],id_nouv_noeud,facette1[2]]

        # tri2 --> tri21 et tri22
        facette2 = self.liste_cells[id_cell].liste_tri3[tri2]
        while((facette2[0] == id1) or (facette2[0] == id2)):
            facette2 = np.roll(facette2,1)
        f21 = [facette2[0],facette2[1],id_nouv_noeud]
        f22 = [facette2[0],id_nouv_noeud,facette2[2]]                
        
        # nouveaux elements
        id_nouv_1 = self.ajoute_elem_ij_famille(f11[1],f11[2],0)
        id_nouv_2 = self.ajoute_elem_ij_famille(f12[1],f12[2],0)
        id_nouv_11 = self.ajoute_elem_ij_famille(f12[0],f12[1],0)
        id_nouv_22 = self.ajoute_elem_ij_famille(f22[0],f22[1],0)

        # ajustement llib
        llib_arete = self.liste_elements[id_arete][3]
        self.liste_elements[id_nouv_1][3] = llib_arete/2.
        self.liste_elements[id_nouv_2][3] = llib_arete/2.
        self.liste_elements[id_nouv_11][3] *= 0.9
        self.liste_elements[id_nouv_22][3] *= 0.9

        # nouvelles facettes
        self.liste_cells[id_cell].liste_tri3.append(f11)
        self.liste_cells[id_cell].liste_tri3.append(f12)
        self.liste_cells[id_cell].liste_tri3.append(f21)
        self.liste_cells[id_cell].liste_tri3.append(f22)
        
        # supprime facettes 2 puis 1
        self.liste_cells[id_cell].liste_tri3.pop(tri2)
        self.liste_cells[id_cell].liste_tri3.pop(tri1)
        
        # supprime element
        self.supprime_element(id_arete)
    
    def cell_division_arete2(self, id_cell, id_arete):
        """
        Division de l'arete id_arete appartenant à la cellule id_cell
        Autre algo - on remplace les facettes par 3 triangles (plus simple)
        
        Args:
            id_cell (_type_): cellule concernee
            id_arete (_type_): arete concernee
        """
        # identification facettes concernées
        facettes = []
        for id_tri, tri in enumerate(self.liste_cells[id_cell].liste_tri3):
            for ij in [[0,1],[1,2],[2,0]]:
                id_eli = self.index_element_ij(tri[ij[0]],tri[ij[1]])
                if id_eli == id_arete:
                    facettes.append(id_tri)
                    #print(f"facette {id_tri} concernée")
        
        tri1 = min(facettes)
        tri2 = max(facettes)
        
        # nouveau noeud tri1 puis nouvelles faces
        face1 = self.liste_cells[id_cell].liste_tri3[tri1]
        pt11 = self.pos_noeuds[face1[0]]
        pt12 = self.pos_noeuds[face1[1]]
        pt13 = self.pos_noeuds[face1[2]]
        pt_nouv_noeud_1 = (np.array(pt11) + np.array(pt12) + np.array(pt13))*(1./3.)
        id_nouv_noeud_1 = self.ajoute_noeud_famille(pt_nouv_noeud_1,0)
        f11 = [face1[0],face1[1],id_nouv_noeud_1]
        f12 = [face1[1],face1[2],id_nouv_noeud_1]
        f13 = [face1[2],face1[0],id_nouv_noeud_1]

        # nouveau noeud tri2 puis nouvelles faces
        face2 = self.liste_cells[id_cell].liste_tri3[tri2]
        pt21 = self.pos_noeuds[face2[0]]
        pt22 = self.pos_noeuds[face2[1]]
        pt23 = self.pos_noeuds[face2[2]]
        pt_nouv_noeud_2 = (np.array(pt21) + np.array(pt22) + np.array(pt23))*(1./3.)
        id_nouv_noeud_2 = self.ajoute_noeud_famille(pt_nouv_noeud_2,0)
        f21 = [face2[0],face2[1],id_nouv_noeud_2]
        f22 = [face2[1],face2[2],id_nouv_noeud_2]
        f23 = [face2[2],face2[0],id_nouv_noeud_2]
        
        # nouveaux elements
        id_nouv_11 = self.ajoute_elem_ij_famille(f11[1],f11[2],0)
        id_nouv_12 = self.ajoute_elem_ij_famille(f12[1],f12[2],0)
        id_nouv_13 = self.ajoute_elem_ij_famille(f13[1],f13[2],0)
        
        id_nouv_21 = self.ajoute_elem_ij_famille(f21[1],f21[2],0)
        id_nouv_22 = self.ajoute_elem_ij_famille(f22[1],f22[2],0)
        id_nouv_23 = self.ajoute_elem_ij_famille(f23[1],f23[2],0)

        # ajustement llib
        # llib_arete = self.liste_elements[id_arete][3]
        # self.liste_elements[id_nouv_1][3] = llib_arete/2.
        # self.liste_elements[id_nouv_2][3] = llib_arete/2.
        # self.liste_elements[id_nouv_11][3] *= 0.9
        # self.liste_elements[id_nouv_22][3] *= 0.9

        # nouvelles facettes
        self.liste_cells[id_cell].liste_tri3.append(f11)
        self.liste_cells[id_cell].liste_tri3.append(f12)
        self.liste_cells[id_cell].liste_tri3.append(f13)
        self.liste_cells[id_cell].liste_tri3.append(f21)
        self.liste_cells[id_cell].liste_tri3.append(f22)
        self.liste_cells[id_cell].liste_tri3.append(f23)
        
        # supprime facettes 2 puis 1
        self.liste_cells[id_cell].liste_tri3.pop(tri2)
        self.liste_cells[id_cell].liste_tri3.pop(tri1)
            
    def cell_uniformise_llib(self,id_cell,l0):
        """
        Uniformise les llib des elements de la cellule

        Args:
            id_cell (_type_): id de la cellule
        """
        cell = self.liste_cells[id_cell]
        for tri in cell.liste_tri3:
            for ij in [[0,1],[1,2],[2,0]]:
                idel = self.index_element_ij(tri[ij[0]],tri[ij[1]])
                self.liste_elements[idel][3] = l0            
    
    def cell_calc_volume(self,id_cell):
        """
        Calcule le volume de la cellule id_cell

        Args:
            id_cell (_type_): id of the cell
        """
        vol = 0
        cell = self.liste_cells[id_cell]
        id_p0 = cell.liste_tri3[0][0]
        pt0 = np.array(self.pos_noeuds[id_p0])
        for tri in cell.liste_tri3:
            pt1 = np.array(self.pos_noeuds[tri[0]])
            pt2 = np.array(self.pos_noeuds[tri[1]])
            pt3 = np.array(self.pos_noeuds[tri[2]])
            v12 = pt2 - pt1
            v13 = pt3 - pt1
            v01 = pt1 - pt0
            vn = np.cross(v12,v13)
            norme_vn = norme(vn)
            base = 0.5*norme_vn
            hauteur = np.dot(vn,v01)/norme_vn
            vol += base * hauteur/3
        cell.volume = vol
        
    def ajoute_cell_base(self, pt0, l0):
        """
        Ajoute une cellule de base = un tetraedre simple
        constitué de coté l0 à partir du point pt0

        Args:
            pt0 (Vec3): point d'insertion
            l0 (float): coté
        """
        pt1 = pt0 + np.array([l0,0,0])
        pt2 = pt0 + np.array([0,l0,0])
        pt3 = pt0 + np.array([0,0,l0])
        fam_no = 0
        fam_el = 0
        id0 = self.ajoute_noeud_famille(pt0,fam_no)
        id1 = self.ajoute_noeud_famille(pt1,fam_no)
        id2 = self.ajoute_noeud_famille(pt2,fam_no)
        id3 = self.ajoute_noeud_famille(pt3,fam_no)
        ide0 = self.ajoute_elem_ij_famille(id0,id1,fam_el)
        ide1 = self.ajoute_elem_ij_famille(id1,id2,fam_el)
        ide2 = self.ajoute_elem_ij_famille(id2,id0,fam_el)
        ide3 = self.ajoute_elem_ij_famille(id0,id3,fam_el)
        ide4 = self.ajoute_elem_ij_famille(id1,id3,fam_el)
        ide5 = self.ajoute_elem_ij_famille(id2,id3,fam_el)
        
        cell0 = Cell()
        cell0.liste_tri3 = [[id0,id2,id1],[id0,id1,id3],[id1,id2,id3],[id0,id3,id2]]
        self.liste_cells.append(cell0)
        return len(self.liste_cells) - 1
    
    def supprime_element(self, id_el):
        """
        Supprime l'élement concerné

        Args:
            id_el (_type_): index de l'element
        """
        # traitement de liste_elements
        self.liste_elements.pop(id_el)
        
        # traitement de liste_elements_bloques
        id_pop = -1
        for i,idsel in enumerate(self.liste_elements_selectionnes):
            if idsel == id_el:
                id_pop = i
            if idsel > id_el:
                self.liste_elements_selectionnes[i] -= 1
        if id_pop >= 0:
            self.liste_elements_selectionnes.pop(id_pop)
        
    def supprime_noeud(self, id_no):
        """
        Supprime le noeud id_no

        Args:
            id_no (_type_): index du noeud
        """
        # traitement pos_noeud, vit_noeud
        self.pos_noeuds.pop(id_no)
        self.vit_noeuds.pop(id_no)
        
        # traitement self.liste_noeuds_bloques
        id_pop = -1
        for i,idi in enumerate(self.liste_noeuds_bloques):
            if idi == id_no:
                id_pop = i
            if idi > id_no:
                self.liste_noeuds_bloques[i] -= 1
        if id_pop >= 0:
            self.liste_noeuds_bloques.pop(id_pop)
            
        # traitement self.liste_noeuds_selectionnes
        id_pop = -1
        for i,idi in enumerate(self.liste_noeuds_selectionnes):
            if idi == id_no:
                id_pop = i
            if idi > id_no:
                self.liste_noeuds_selectionnes[i] -= 1
        if id_pop >= 0:
            self.liste_noeuds_selectionnes.pop(id_pop)
        
        # traitement self.liste_cells
        #      decalage indices seulement
        #      il conviendrait de verifier les elements aussi
        for cell in self.liste_cells:
            for tri in cell.liste_tri3:
                for i in range(3):
                    if tri[i] > id_no:
                        tri[i] -= 1
        
    def ajoute_noeud_famille(self, vec3, fam):
        """
        ajoute un noeud de coordonnees vec3 de la famille donnee
        """
        ind_retour = self.index_noeud(vec3)
        if ind_retour == -1:
            self.pos_noeuds.append(vec3)
            self.fam_noeuds.append(fam)
            self.vit_noeuds.append([0., 0., 0.])
            ind_retour = len(self.pos_noeuds) - 1
        else:
            # le noeud existe mais on lui attribue la nouvelle famille
            #print("noeud existe...")
            self.fam_noeuds[ind_retour] = fam
        return ind_retour
    
    def ajoute_elem_ij_famille(self, i, j, fam):
        """
        ajoute un element entre le noeud i et noeud j de famille fam
        """
        #print("ajoute elem", i, j, fam)
        nb_noeuds = len(self.pos_noeuds)
        #print("nb_noeuds", nb_noeuds)
        if (i >= 0) and (i < nb_noeuds) and (j >= 0) and (j < nb_noeuds) and (i != j):
            #print("pour l'instant ca va")
            #on peut y aller
            ind_existe_deja = self.index_element_ij(i,j)
            existe_deja = False
            if ind_existe_deja >=0 and self.liste_elements[ind_existe_deja][2] == fam:
                #print("en plus il est de la meme famille")
                existe_deja = True
            if (not existe_deja) and (fam >= 0) and (fam < len(self.liste_car_elements)):
                # on peut l'ajouter
                # calcul de la longueur
                vdir = np.array(self.pos_noeuds[i]) - np.array(self.pos_noeuds[j])
                self.liste_elements.append([i, j, fam, norme(vdir)])
                ind_existe_deja = len(self.liste_elements) - 1
            return ind_existe_deja

    def ajoute_elem_p1p2_fn1_fn2_felem(self, pt1, pt2, fn1, fn2, fel):
        """
        ajoute un element entre les noeuds pt1 et pt2
        S'occupe de verifier leur existence d'abord
        """
        ind1 = self.ajoute_noeud_famille(pt1, fn1)
        ind2 = self.ajoute_noeud_famille(pt2, fn2)
        ind_retour = self.ajoute_elem_ij_famille(ind1, ind2, fel)
        return ind_retour

    def bloque_noeud(self, indice_noeud, bx, by, bz):
        """
        bloque le noeud selon bx,by,bz
        """
        if indice_noeud >= 0 and indice_noeud < len(self.pos_noeuds):
            # indice_noeud compatible, on peut y aller
            index_noeud = -1
            for ind, no in enumerate(self.liste_noeuds_bloques):
                if no[0] == indice_noeud:
                    index_noeud = ind
            if index_noeud == -1:
                self.liste_noeuds_bloques.append([indice_noeud, bx, by, bz])
            else:
                self.liste_noeuds_bloques[index_noeud] = [indice_noeud, bx, by, bz]

    def index_element_ij(self, ind1, ind2):
        """
        Renvoie l'index d'un element geometrique (pt1,pt2), s'il existe
        """
        ind_retour = -1
        for ind0,el in enumerate(self.liste_elements):
            if ((el[0] == ind1 and el[1] == ind2) or (el[1] == ind1 and el[0] == ind2)):
                ind_retour = ind0
        # if ind_retour >= 0 :
        #     print("on a identifie un element")
        return ind_retour

    def index_noeud(self, vec3, tol=1.E-6):
        """
        renvoie l'index du noeud situé au point vec3, à une tolérance près
        """
        index_retour = -1
        # lent
        for ind, no in enumerate(self.pos_noeuds):
            if norme_au_carre(np.array(vec3) - np.array(no)) < tol * tol:
                index_retour = ind
        return index_retour
    
    def regle_carac_famille_noeud(self,fam,rayon,masse,rigidite,couleur):
        """
        reglage des caracteristiques de la famille de noeud fam
        """
        self.liste_car_noeud[fam]["rayon"] = rayon
        self.liste_car_noeud[fam]["masse"] = masse
        self.liste_car_noeud[fam]["rigidite"] = rigidite
        self.liste_car_noeud[fam]["couleur"] = couleur

    def regle_carac_famille_elements(self,fam,rayon,masse_lin,rig_EA,couleur):
        """
        reglage des caracteristiques de la famille d'elem fam
        """
        self.liste_car_elements[fam]["rayon"] = rayon
        self.liste_car_elements[fam]["masse_lin"] = masse_lin
        self.liste_car_elements[fam]["rig_EA"] = rig_EA
        self.liste_car_elements[fam]["couleur"] = couleur
    
    def dessinGL_cells(self):
        """
        Dessine les cellules en triangle
        """
        for cell in self.liste_cells:
            for tri in cell.liste_tri3:
                pt1 = self.pos_noeuds[tri[0]]
                pt2 = self.pos_noeuds[tri[1]]
                pt3 = self.pos_noeuds[tri[2]]
                glColor4f(0, 1, 0, 0.5)
                dessin_triangle(pt1,pt2,pt3)
    
    def dessinGL_noeuds(self):
        """
        Dessine les elements sous forme de sphere
        """
        for ind, pos in enumerate(self.pos_noeuds):
            carac = self.liste_car_noeud[self.fam_noeuds[ind]]
            rayon = carac["rayon"]
            couleur = carac["couleur"]
            if ind in self.liste_noeuds_selectionnes:
                glColor3f(1, 1, 0)
                rayon *= 1.1
            else:
                glColor3f(couleur[0], couleur[1], couleur[2])
            if ind in self.liste_noeuds_contact:
                glColor3f(1, 0, 0)
                rayon *= 1.5
            glLoadName(ind)
            dessin_sphere(pos, rayon)
                
    def dessinGL_elements_lignes(self):
        """
        Dessine les elements sous forme de ligne
        """
        for el in self.liste_elements:
            no_deb = self.pos_noeuds[el[0]]
            no_fin = self.pos_noeuds[el[1]]
            glBegin(GL_LINES)
            glVertex3f(no_deb[0], no_deb[1], no_deb[2])
            glVertex3f(no_fin[0], no_fin[1], no_fin[2])
            glEnd()
            #dessin_cylindre(no_deb,no_fin,1)

    def dessinGL_elements_tubes(self):
        """
        Dessine les elements sous forme de tube
        """
        for ind,el in enumerate(self.liste_elements):
            no_deb = self.pos_noeuds[el[0]]
            no_fin = self.pos_noeuds[el[1]]
            #rayon1 = self.liste_car_noeud[self.fam_noeuds[el[0]]]["rayon"]
            #rayon2 = self.liste_car_noeud[self.fam_noeuds[el[1]]]["rayon"]
            rayon_elem = self.liste_car_elements[el[2]]["rayon"]
            couleur = self.liste_car_elements[el[2]]["couleur"]
            if ind in self.liste_elements_selectionnes:
                glColor3f(1, 1, 0)
                rayon_elem *= 1.1
            else:
                glColor3f(couleur[0], couleur[1], couleur[2])
            glLoadName(ind)
            dessin_cylindre(no_deb, no_fin, rayon_elem, rayon_elem)
    
    def dessinGL_blocages_noeuds(self):
        """
        Dessine un cone noir selon chaque axe bloque pour tous les noeuds bloques
        """
        glColor3f(0.,0.,0.)
        glDisable(GL_CULL_FACE)
        for nob in self.liste_noeuds_bloques:
            pt = self.pos_noeuds[nob[0]]
            rayon = self.liste_car_noeud[self.fam_noeuds[nob[0]]]["rayon"]
            if nob[1] > 0:
                # dessine un cone selon x
                glPushMatrix()
                glTranslatef(pt[0]-2*rayon,pt[1],pt[2])
                glRotatef(90.,0.,1.,0.)
                glutSolidCone(rayon,rayon,10,10)
                glPopMatrix()
            if nob[2] > 0:
                # dessine un cone selon y
                glPushMatrix()
                glTranslatef(pt[0],pt[1]-2*rayon,pt[2])
                glRotatef(-90.,1.,0.,0.)
                glutSolidCone(rayon,rayon,10,10)
                glPopMatrix()
                
            if nob[3] > 0:
                # dessine un cone selon z
                glPushMatrix()
                glTranslatef(pt[0],pt[1],pt[2]-2*rayon)
                #glRotatef(90.,0.,.,1.)
                glutSolidCone(rayon,rayon,10,10)
                glPopMatrix()
        glDisable(GL_CULL_FACE)
    
    def evolution_dynamique(self, t, dt):
        #pos0 = np.array(self.pos_noeuds).copy()
        etat_actuel = np.array([self.pos_noeuds, self.vit_noeuds]).copy()
        etat_suivant = self.RKOneD(etat_actuel, t, dt)
        self.pos_noeuds = list(etat_suivant.copy()[0])
        self.vit_noeuds = list(etat_suivant.copy()[1])
        
        if flags["relaxation_dynamique"]:
            self.Ecin.append(np.square(self.vit_noeuds).sum())
            #print(self.vit_noeuds)
            if len(self.Ecin) > 3:
                #print(self.Ecin)
                self.Ecin.pop(0)
                if (self.Ecin[1] > self.Ecin[0]) and (self.Ecin[1] > self.Ecin[0]):
                    for vit in self.vit_noeuds:
                        vit *= 0.
    
    # integrateurs temporel
    def RKOneD(self, x, t, dt):
        """
        RungeKutta ordre 4. necessite des etats de type np.array
        """
        k1 = dt * self.derive_etat(x, t)
        k2 = dt * self.derive_etat(x + k1 / 2.0, t)
        k3 = dt * self.derive_etat(x + k2 / 2.0, t)
        k4 = dt * self.derive_etat(x + k3, t)
        etat_tpdt = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return etat_tpdt

    def EulerOneD(self, x, t, dt):
        """
        Euler 1D
        """
        etat_tpdt = x + dt * self.derive_etat(x, t)
        return etat_tpdt

    def derive_etat(self, etat, t):
        """
        Derivation de l'etat du systeme, donne sous la forme (pos,vit)
        En dynamique, le systeme derive est (vit,acc), avec acc = F/m
        On en revient donc au calcul des forces appliquées aux noeuds
        """
        etat2 = etat.copy()*0.
        forces_noeuds = etat2[0].copy()
        

        nb_noeuds = len(etat[0])
        
        # calcul des forces aux noeuds
        for ind in range(nb_noeuds):
            # contact sol
            car_no = self.liste_car_noeud[self.fam_noeuds[ind]]
            dz = etat[0][ind][2] - car_no['rayon']
            if dz < 0.:
                forces_noeuds[ind] -= np.array([0., 0., car_no['rigidite'] * dz])
                
            # gravite
            forces_noeuds[ind] += np.array([0., 0., params["gravite"] * car_no['masse']])
            
            #amortissement visqueux
            forces_noeuds[ind] -= np.array(etat[1][ind]) * params["viscosite_milieu"]
        
        #efforts dus aux elements
        self.liste_f_elements = [[0,0,0]] * len(self.liste_elements)
        for ind, el in enumerate(self.liste_elements):
            car_el = self.liste_car_elements[el[2]]
            ind0 = el[0]
            ind1 = el[1]
            l0 = el[3]
            v_elem = np.array(etat[0][ind1]) - np.array(etat[0][ind0])
            l_elem = norme(v_elem)
            v_force = v_elem * car_el['rig_EA'] * (l_elem - l0) / (l0 * l_elem)
            self.liste_f_elements[ind] = v_force
            forces_noeuds[ind0] += v_force
            forces_noeuds[ind1] -= v_force
        
        # efforts de pression
        for id_cell,cell in enumerate(self.liste_cells):
            # calcul volume
            self.cell_calc_volume(id_cell)
            #print(f"Le volume de la cellule {id_cell} est {cell.volume}")
            
            # calcul pression
            v0 = cell.volume0
            cell.pression =(cell.volume0 - cell.volume)*cell.dp_dv
            #print(f"           et sa pression est {cell.pression}")
            
            # application aux noeuds
            for tri in cell.liste_tri3:
                pt0 = np.array(self.pos_noeuds[tri[0]])
                pt1 = np.array(self.pos_noeuds[tri[1]])
                pt2 = np.array(self.pos_noeuds[tri[2]])
                v01 = pt1 - pt0
                v02 = pt2 - pt0
                vn = np.cross(v01,v02)
                #vn_norme = norme(vn)
                v_force = cell.pression * vn / 2
                forces_noeuds[tri[0]] += v_force/3
                forces_noeuds[tri[1]] += v_force/3
                forces_noeuds[tri[2]] += v_force/3
                
        #contact noeud-noeud
        if flags["contact_noeud_noeud"]:
            for i in range(nb_noeuds - 1):
                fam_i = self.fam_noeuds[i]
                car_i = self.liste_car_noeud[fam_i]
                r_i = car_i['rayon']
                for j in range(i + 1, nb_noeuds):
                    fam_j = self.fam_noeuds[j]
                    car_j = self.liste_car_noeud[fam_j]
                    r_j = car_j['rayon']
                    dmin = r_i + r_j
                    vdir = etat[0][j] - etat[0][i]
                    if not(fam_i == 1 and fam_j == 1):
                        if abs(vdir[0]) < dmin:
                            if abs(vdir[1]) < dmin:
                                if abs(vdir[2]) < dmin:
                                    distsq = sum(vdir * vdir)
                                    gap = distsq - (r_i + r_j) * (r_i + r_j)
                                    if gap < 0:
                                        #contact
                                        k_i = car_i['rigidite']
                                        k_j = car_j['rigidite']
                                        dist = np.sqrt(distsq)
                                        force = (1 / (1 / k_i + 1 / k_j)) * (dist - np.sqrt((r_i + r_j) * (r_i + r_j)))
                                        forces_noeuds[i] += force * vdir / dist
                                        forces_noeuds[j] -= force * vdir / dist
        
        # contact cell-cell
        tcc0 = time.time()
        if (flags["contact-cell-cell"]) and t > 0.5:
            self.liste_noeuds_contact = []
            r_no = self.liste_car_noeud[0]["rayon"]
            k_no = self.liste_car_noeud[0]["rigidite"]
            nb_cells = len(self.liste_cells)
            for i in range(nb_cells-1):
                cell_i = self.liste_cells[i]
                for j in range(i+1,nb_cells):
                    cell_j = self.liste_cells[j]
                    for tri3_i in cell_i.liste_tri3:
                        ptsi = [self.pos_noeuds[i] for i in tri3_i]
                        cdg_i = 0.333333*sum(ptsi)
                        for tri3_j in cell_j.liste_tri3:
                            ptsj = [self.pos_noeuds[i] for i in tri3_j]
                            cdg_j = 0.333333*sum(ptsj)
                            dij = cdg_j - cdg_i
                            lij2 = norme_au_carre(dij)
                            if lij2 < 1:
                                # contact possible
                                # # traitement noeud i
                                # for ii,pti in enumerate(ptsi):
                                #     vgipi = pti - cdg_i
                                #     fpi = -kcc * dij * max(np.dot(vgipi,dij),0) / lij2
                                #     forces_noeuds[tri3_i[ii]] += fpi
                                # for jj,ptj in enumerate(ptsj):
                                #     vgjpj = ptj - cdg_j
                                #     fpj = -kcc * dij * min(np.dot(vgjpj,dij),0) / lij2
                                #     forces_noeuds[tri3_j[jj]] += fpj
                                
                                # noeuds du triangle i avec triangle j
                                for ii,pti in enumerate(ptsi):
                                    alpha_i,beta_i,gamma_i,vn_j = alpha_beta_gamma_vn_tri_pt(ptsj[0],ptsj[1],ptsj[2],pti)
                                    if gamma_i > -1.5:
                                        if gamma_i < r_no*1.5:
                                            if alpha_i >= 0:
                                                if beta_i >= 0:
                                                    if alpha_i + beta_i <= 1:
                                                        #print(f"contact possible avec {alpha_i}{beta_i}{gamma_i}")
                                                        #self.liste_noeuds_contact.append((tri3_i[ii],tri3_j))
                                                        
                                                        # gestion du contact - calcul force de contact et resultantes
                                                        R = (gamma_i - r_no)*k_no
                                                        RA = R * (1 - alpha_i - beta_i)
                                                        RB = R * alpha_i
                                                        RC = R * beta_i
                                                        
                                                        # application des forces
                                                        forces_noeuds[tri3_j[0]] += vn_j*RA
                                                        forces_noeuds[tri3_j[1]] += vn_j*RB
                                                        forces_noeuds[tri3_j[2]] += vn_j*RC
                                                        forces_noeuds[tri3_i[ii]] -= vn_j*R
                                # noeuds du triangle j avec triangle i
                                for jj,ptj in enumerate(ptsj):
                                    alpha_j,beta_j,gamma_j,vn_i = alpha_beta_gamma_vn_tri_pt(ptsi[0],ptsi[1],ptsi[2],ptj)
                                    if gamma_j > -1.5:
                                        if gamma_j < r_no*1.5:
                                            if alpha_j >= 0:
                                                if beta_j >= 0:
                                                    if alpha_j + beta_j <= 1:
                                                        #print(f"contact possible avec {alpha_j}{beta_j}{gamma_j}")
                                                        #self.liste_noeuds_contact.append((tri3_j[jj],tri3_i))
                                                        # gestion du contact - calcul force de contact et resultantes
                                                        R = (gamma_j - r_no)*k_no
                                                        RA = R * (1 - alpha_j - beta_j)
                                                        RB = R * alpha_j
                                                        RC = R * beta_j
                                                        
                                                        # application des forces
                                                        forces_noeuds[tri3_i[0]] += vn_i*RA
                                                        forces_noeuds[tri3_i[1]] += vn_i*RB
                                                        forces_noeuds[tri3_i[2]] += vn_i*RC
                                                        forces_noeuds[tri3_j[jj]] -= vn_i*R
            #if len(self.liste_noeuds_contact) > 0:
            #    print(f"Liste noeuds contact = {self.liste_noeuds_contact}")
        tcc1 = time.time()
        global delta_time_ccc
        delta_time_ccc = tcc1 - tcc0
               
        ########################################################################
        # calcul de l'etat derive
        for ind in range(len(etat[0])):
            masse_no = self.liste_car_noeud[self.fam_noeuds[ind]]['masse']
            etat2[1][ind] = forces_noeuds[ind] / masse_no
            etat2[0][ind] = etat[1][ind].copy()
            
        # prise en compte des noeuds bloques
        for nob in self.liste_noeuds_bloques:
            for j in range(3):
                if nob[j+1] != 0:
                    etat2[0][nob[0]][j] = 0 
        
        return etat2
    
    def affiche_stats(self):
        """
        Affiche des statistiques sur l'etat actuel
        """        
        # position du noeud selectionne
        if len(self.liste_noeuds_selectionnes) == 1:
            no_sel = self.liste_noeuds_selectionnes[0]
            pt_no_sel = self.pos_noeuds[no_sel]
            print("Noeud ",no_sel, pt_no_sel)

        # L, Llib et effort dans l'elem selectionne
        if len(self.liste_elements_selectionnes) == 1:
            el_sel = self.liste_elements_selectionnes[0]
            el = self.liste_elements[el_sel]
            pt1 = self.pos_noeuds[el[0]]
            pt2 = self.pos_noeuds[el[1]]
            fam = el[2]
            llib = el[3]
            car = self.liste_car_elements[fam]
            l = norme(np.array(pt2)-np.array(pt1))
            n = car["rig_EA"]*(l-llib)/llib
            print("Element ",el_sel," - L = ",l," - Llib = ",llib," - T = ",n)
           
        # pressions des cellules
        for i_cell, celli in enumerate(self.liste_cells):
            print(f" - cellule {i_cell} - pression={celli.pression} - volume={celli.volume} - volume0={celli.volume0}")
 
# ------------------------------------------------------------------------------


def setCamera():
    #print " camera :"
    #print "   dist : ", distance
    #print "   elev : ", elev
    #print "   azim : ", azim
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0., 0., -distance)
    glRotatef(elev, 1., 0., 0.)
    glRotatef(azim, 0., 0., 1.)
    glTranslatef(origine[0], origine[1], origine[2])

def dessin_cylindre(pt1, pt2, rayon1, rayon2):
    """
    Dessine un cylindre entre le pt1 et le pt2
    """
    dpt = np.array(pt2) - np.array(pt1)
    v = norme(dpt)
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

def dessin_sphere(pos, rayon):
    """
    Dessine une sphere à la position indiquee, du rayon indiqué
    (10 subdivisions par defauts)
    """
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glTranslatef(pos[0], pos[1], pos[2])
    glutSolidSphere(rayon, 10, 5)
    glPopMatrix()
        
def dessin_repere():
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

def dessin_triangle(pt1,pt2,pt3):
    glBegin(GL_TRIANGLES)
    glVertex3f(pt1[0], pt1[1], pt1[2])
    glVertex3f(pt2[0], pt2[1], pt2[2])
    glVertex3f(pt3[0], pt3[1], pt3[2])
    glEnd()
    
    # et la normale en plus
    v0 = (np.array(pt1) + np.array(pt2) + np.array(pt3))/3
    vn = np.cross(np.array(pt2-pt1),np.array(pt3-pt1))
    vn = vn / norme(vn)
    v1 = v0 + vn
    glBegin(GL_LINES)
    glVertex3f(v0[0], v0[1], v0[2])
    glVertex3f(v1[0], v1[1], v1[2])
    glEnd()
    

def dessin_plan():
    """
    Dessin du plan de base
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
    mon_sys.dessinGL_noeuds()
    mon_sys.dessinGL_elements_tubes()
    mon_sys.dessinGL_blocages_noeuds()
    dessin_repere()
    dessin_plan()
    
    mon_sys.dessinGL_cells()

    #########################################
    
    # basculement buffer : affichage
    glutSwapBuffers()

    #timing
    previousTime = currentTime
    frameCounter += 1
    if currentTime - frameCounterTimer > 1:
        # affichage toutes les secondes
        print("FPS:", frameCounter)
        print(f"t = {t} - delta_ccc = {delta_time_ccc}")
        frameCounter = 0
        frameCounterTimer = currentTime
        mon_sys.affiche_stats()

def resize(width, height):
    """
    Redimensionnement de la fenetre
    """
    if(height == 0): height = 1
    screenSize = [width, height]
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(80.0, float(width) / float(height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)

def key(*args):
    """
    Gestion du clavier
    """
    print(f"fonction Key appelee avec les arguments {args}")

    if args[0] == b'w':
        params['gravite'] += 5
        print(f"{params}")
    elif args[0] == b'W':
        params['gravite'] -= 5
        print(f"{params}")
    elif args[0] == b'r':
        print("relaxation dynamiqe")
        if flags["relaxation_dynamique"]:
            flags["relaxation_dynamique"] = False
        else:
            flags["relaxation_dynamique"] = True
        print(f"{flags}")
    elif args[0] == b'c':
        print("contact-cell-cell")
        if flags["contact-cell-cell"]:
            flags["contact-cell-cell"] = False
        else:
            flags["contact-cell-cell"] = True
        print(f"{flags}")
    elif args[0] == b'v':
        params["viscosite_milieu"] += 1
        print(f"{params}")
    elif args[0] == b'V':
        params["viscosite_milieu"] -= 1
        print(f"{params}")
    elif args[0] == b'p':
        mon_sys.liste_cells[0].volume0 += 1
        print(f"Volume0 = {mon_sys.liste_cells[0].volume0}")
    elif args[0] == b'P':
        mon_sys.liste_cells[0].volume0 -= 1
        print(f"Volume0 = {mon_sys.liste_cells[0].volume0}")
    elif args[0] == b'\033': #escape
        glutDestroyWindow(window)
        #sys.exit()
        
def mouse(*args):
    """
    Gestion de la souris
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
                selection(mon_sys,x,y,0,glutGetModifiers())
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

def selection(systeme,x,y,type_sel,modificateur):
    """
    gere le clic sur les elements ou noeuds
    """
    no_selected,no_dist = selectionne(x,y,systeme.dessinGL_noeuds)
    el_selected,el_dist = selectionne(x,y,systeme.dessinGL_elements_tubes)
    #print "no_selected =", no_selected
    #print "el_selected =", el_selected
    
    if (no_selected >= 0 or el_selected >= 0):
        if (no_dist < el_dist):
            # le noeud est le plus proche
            systeme.liste_elements_selectionnes = []
            if modificateur == GLUT_ACTIVE_SHIFT:
                # on fait une selection multiple
                if systeme.liste_noeuds_selectionnes.count(no_selected) == 0:
                    systeme.liste_noeuds_selectionnes.append(no_selected)
                else:
                    systeme.liste_noeuds_selectionnes.remove(no_selected)
            else:
                systeme.liste_noeuds_selectionnes = [no_selected]
        else:
            # l'element est le plus proche
            systeme.liste_noeuds_selectionnes = []
            if modificateur == GLUT_ACTIVE_SHIFT:
                # on fait une selection multiple
                if systeme.liste_elements_selectionnes.count(el_selected) == 0:
                    systeme.liste_elements_selectionnes.append(el_selected)
                else:
                    systeme.liste_elements_selectionnes.remove(el_selected)
            else:
                systeme.liste_elements_selectionnes = [el_selected]
    else:
        print("rien de selectionne...")
    
def selectionne(x,y,dessine):
    """
    renvoie le numero du noeud du systeme sous le point x,y
    """
    buffsize=512
    viewport = glGetIntegerv(GL_VIEWPORT)
    glSelectBuffer(buffsize)
    glRenderMode(GL_SELECT)
    glInitNames()
    glPushName(-1)

    # reglage de la matrice de projection restrictive
    glMatrixMode (GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    #  create 3x3 pixel picking region near cursor location
    gluPickMatrix(x,viewport[3]-y, 5.0, 5.0, viewport)
    
    # reglage perspective et camera
    gluPerspective(80.,(1.*viewport[2]-1.*viewport[0])/(1.*viewport[3]-1.*viewport[1]),.1,100.)
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
    #print "--------- pick3d ----------"
    #print " - nhits =", len(hits )
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
    #print "selection brute = ", selectionne
    return selectionne,min_depth
        
def motion(*args):
    """
    Gestion du mouvement de souris
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
    animation du systeme
    """
    global t, dt, t_div_elem
    # boucle pour faire plusieurs calculs entre deux affichages
    for i in range(1):
        t += dt    # avancement du temps
        #mon_sys.pos_noeuds[10][0] += 0.01
        mon_sys.evolution_dynamique(t, dt)  # evoluton dynamique du systeme
        # on peu rajouter ici une autre fonction d'evolution :
        #   inteligence artificielle du systeme
        comportement_cellule(mon_sys,t)
    display()

def comportement_cellule(syst,temps):
    """
    Gestion du comportement des cellules
    """
    global t, dt, t_div_elem
    
    # croissance du volume0 des cellules
    for cell in syst.liste_cells:
        #print(f"le volume de la cellule en cours est {cell.volume}")
        if cell.pression < 1000 and cell.volume0 < 200: # garde fou
            cell.volume0 += 50*dt
            
    # croissance des llib des elements
    id_el_division = -1
    id_cell_division = -1
    llib_el_division = 0
    l_el_division = 0
    for id_cell, cell in enumerate(syst.liste_cells):
        for tri in cell.liste_tri3:
            for ij in [[0,1],[1,2],[2,0]]:
                no_i = tri[ij[0]]
                no_j = tri[ij[1]]
                idel = syst.index_element_ij(no_i,no_j)
                force = norme(syst.liste_f_elements[idel])
                llib_i = syst.liste_elements[idel][3]
                l_i = norme(np.array(syst.pos_noeuds[no_i]) - np.array(syst.pos_noeuds[no_j]))
                #print(f"Element {idel} - force = {force}")
                if force > 200:
                    # allongement
                    if syst.liste_elements[idel][3] < 1.2: # garde fou
                        syst.liste_elements[idel][3] += 0.005
                if force < 100:
                    # raccourcissement
                    if syst.liste_elements[idel][3] < 0.5: # garde fou
                        syst.liste_elements[idel][3] -= 0.005
                # division
                if l_i > 1.5:
                    if cell.volume < 200:
                        #if force > 500:
                        # uniquement si le plus long
                        #llib_i = syst.liste_elements[idel][3]
                        #if llib_i > llib_el_division:
                        if l_i > l_el_division:
                            id_el_division = idel
                            id_cell_division = id_cell
                            llib_el_division = llib_i
                            l_el_division = l_i
    
    if id_el_division >= 0:
        if t > t_div_elem + 10*dt: # garde fou
            # division de l'élément id_el_division
            #print(f"Division de l'élément {id_el_division} dans la cellule {id_cell_division}")
            syst.cell_division_arete(id_cell_division,id_el_division)
            t_div_elem = t
            #print("on vient de diviser")
    
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
    glEnable( GL_BLEND );

    # reponse aux evenements
    glutDisplayFunc(display)
    glutIdleFunc(animation)
    glutReshapeFunc(resize)
    #glutIgnoreKeyRepeat(1)
    glutKeyboardFunc(key)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)

    # on fait un systeme
    mon_sys = Systeme()

    # reglage des carac des familles de noeuds
    # reglage des carac des familles d'elements
    # familles personnalisees
    mon_sys.regle_carac_famille_noeud(0,.2,.2,1000.,(0.2,0.2,0.2))
    mon_sys.regle_carac_famille_noeud(1,0.5,.1,1000.,(1.,0.,0.))
    mon_sys.regle_carac_famille_noeud(2,0.5,.1,1000.,(0.,1.,0.))

    mon_sys.regle_carac_famille_elements(0,0.1,.2,800,(1.,0.,0.))
    mon_sys.regle_carac_famille_elements(1,0.1,.1,1000,(0.,0.,1.))
    mon_sys.regle_carac_famille_elements(2,0.2,.1,100,(1.,0.,1.))

    # systeme de base
    # pt1 = np.array([0,0,0])
    # pt2 = np.array([2,0,0])
    # pt3 = np.array([0,2,0])
    # pt4 = np.array([0,0,2])
    #####mon_sys.ajoute_noeud_famille(pt1,0)
    # mon_sys.ajoute_elem_p1p2_fn1_fn2_felem(pt1,pt2,0,0,0)
    # mon_sys.ajoute_elem_p1p2_fn1_fn2_felem(pt2,pt3,0,0,0)
    # mon_sys.ajoute_elem_p1p2_fn1_fn2_felem(pt3,pt1,0,0,0)
    # mon_sys.ajoute_elem_p1p2_fn1_fn2_felem(pt1,pt4,0,0,0)
    # mon_sys.ajoute_elem_p1p2_fn1_fn2_felem(pt2,pt4,0,0,0)
    # mon_sys.ajoute_elem_p1p2_fn1_fn2_felem(pt3,pt4,0,0,0)
    
    # systeme = cellule de base ---------------------------------------
    #id_cell1 = mon_sys.ajoute_cell_base([-2,-1,2],1)
    #mon_sys.cell_uniformise_llib(id_cell1,2)
    
    #id_cell2 = mon_sys.ajoute_cell_base([2,-1,2],1)
    #mon_sys.cell_uniformise_llib(id_cell2,2)

    id_cell3 = mon_sys.ajoute_cell_base([0,0,2],1)
    mon_sys.cell_uniformise_llib(id_cell3,2)

    id_cell4 = mon_sys.ajoute_cell_base([3,0,20],1)
    mon_sys.cell_uniformise_llib(id_cell4,2)


    print(len(mon_sys.pos_noeuds), " noeuds")
    print(len(mon_sys.liste_elements), " elements")

    # lancement de GLUT
    glutMainLoop()
