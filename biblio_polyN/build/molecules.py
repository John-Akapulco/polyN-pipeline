# -*- coding: utf-8 -*-
"""
Base de donnees des structures polyazotees extraites de la bibliographie :

  [B]  Fau, Tobita, Wilson, Perera, Bartlett - "Structure and Stability of
       Polynitrogen Molecules and Their Spectroscopic Characteristics"
       (Univ. of Florida, QTP), fichier polynitrogen1.pdf
  [GS] Glukhovtsev, Jiao, Schleyer - "Besides N2, What Is the Most Stable
       Molecule Composed Only of Nitrogen Atoms?", Inorg. Chem. 1996, 35,
       7124-7133, fichier poly-N.pdf

Chaque entree decrit soit une construction geometrique EXACTE (a partir des
longueurs de liaison / angles publies + symetrie du groupe ponctuel), soit
une construction "xtb_generic" : la connectivite (topologie) est fixee a
partir de la description de la reference, la geometrie 3D initiale est
generique puis relaxee par optimisation GFN2-xTB (methode utilisee dans le
pipeline polyN_pipeline.py de ce depot) faute d'angles dièdres publies.

build types:
  'linear'            params: bonds (list) ou bond (scalaire), n
  'triangle'          params: edge
  'bent'              params: bond, angle  (2 liaisons identiques)
  'chain2d'           params: bonds[], angles[], turns[] (optionnel)
  'ring_regular'      params: n, edge
  'ring_general'      params: bonds[n], angles[n]  (cyclique)
  'tetrahedron'       params: edge
  'cube'              params: edge
  'prism'             params: n, edge_ring, edge_vertical
  'puckered_4ring'    params: edge, angle
  'trigonal_bipyramid_nocenter' params: bond, angle_apex
  'two_rings_ortho'   params: ring_a (spec imbriquee), ring_b (spec imbriquee), connect_len
  'xtb_generic'       params: n_atoms, shape hint ('ring','chain','cage'), notes
"""

M = []


def add(**kw):
    M.append(kw)


# ---------------------------------------------------------------------
# N2, N2+, N3 family
# ---------------------------------------------------------------------
add(id="N2", formula="N2", charge=0, mult=1, point_group="Dinfh",
    source="[B] p.7 (CCSD(T)/aug-cc-pVTZ)", method="CCSD(T)/aug-cc-pVTZ",
    build="linear", params=dict(bonds=[1.1035]))

add(id="N2+", formula="N2", charge=1, mult=2, point_group="Dinfh",
    source="[B] p.7 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="linear", params=dict(bonds=[1.1166]))

add(id="N3+_linear", formula="N3", charge=1, mult=1, point_group="Dinfh",
    source="[B] p.9 (CCSD(T)/cc-pVTZ)", method="CCSD(T)/cc-pVTZ",
    build="linear", params=dict(bonds=[1.1887, 1.1887]))

add(id="N3+_triangle", formula="N3", charge=1, mult=1, point_group="D3h",
    source="[B] p.11 (CCSD(T)/cc-pVTZ)", method="CCSD(T)/cc-pVTZ",
    build="triangle", params=dict(edge=1.3271))

add(id="N3_radical", formula="N3", charge=0, mult=2, point_group="Dinfh",
    source="[B] p.12 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="linear", params=dict(bonds=[1.1846, 1.1846]))

add(id="N3-_bent", formula="N3", charge=-1, mult=1, point_group="C2v",
    source="[B] p.13 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="bent", params=dict(bond=1.2942, angle=84.76))

add(id="N3-_triangle", formula="N3", charge=-1, mult=3, point_group="D3h",
    source="[B] p.14 (B3LYP/aug-cc-pVDZ, triplet 3A1')", method="B3LYP/aug-cc-pVDZ",
    build="triangle", params=dict(edge=1.4055))

add(id="N3-_linear", formula="N3", charge=-1, mult=1, point_group="Dinfh",
    source="[B] p.14 (CCSD(T)/aug-cc-pVTZ)", method="CCSD(T)/aug-cc-pVTZ",
    build="linear", params=dict(bonds=[1.1897, 1.1897]))

# ---------------------------------------------------------------------
# N4 family
# ---------------------------------------------------------------------
add(id="N4+_rectangle", formula="N4", charge=1, mult=2, point_group="D2h",
    source="[B] p.16 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="rectangle", params=dict(a=1.6314, b=1.2085))

add(id="N4_C2v_butterfly", formula="N4", charge=0, mult=3, point_group="C2v",
    source="[B] p.16-17, table p.73 ('butterfly')", method="CCSD(T)/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=4, shape="ring3+pendant"),
    notes="Cycle a 3 centres (~1.45 A) + azote exocyclique (~1.56 A); "
          "dièdre non publie -> relaxation GFN2-xTB (mult=3).")

add(id="N4_D2d_puckered", formula="N4", charge=0, mult=3, point_group="D2d",
    source="[B] p.18 (CCSD(T)/aug-cc-pVDZ, 3A2)", method="CCSD(T)/aug-cc-pVDZ",
    build="puckered_4ring", params=dict(edge=1.3943, angle=88.39))

add(id="N4_Td", formula="N4", charge=0, mult=1, point_group="Td",
    source="[B] p.19 (CCSD(T)/aug-cc-pVTZ) - tetraazatetrahedrane", method="CCSD(T)/aug-cc-pVTZ",
    build="tetrahedron", params=dict(edge=1.4613))

add(id="N4_D2h_rectangle_planar", formula="N4", charge=0, mult=1, point_group="D2h",
    source="[B] p.20 (B3LYP/aug-cc-pVDZ, non-minimum a plus haut niveau)",
    method="B3LYP/aug-cc-pVDZ",
    build="rectangle", params=dict(a=1.5372, b=1.2562))

add(id="N4_C2h_chain", formula="N4", charge=0, mult=3, point_group="C2h",
    source="[B] p.22 (CCSD(T)/aug-cc-pVDZ, 3Bu, 'linear E')", method="CCSD(T)/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1987, 1.5292, 1.1987], angles=[113.15, 113.15], turns=[1, -1]))

add(id="N4-_rectangle", formula="N4", charge=-1, mult=2, point_group="D2h",
    source="[B] p.23 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="rectangle", params=dict(a=1.4524, b=1.3136))

add(id="N4-_chain_planar", formula="N4", charge=-1, mult=4, point_group="C2h",
    source="[B] p.23 (B3LYP/aug-cc-pVDZ, 4Bg, 'planar')", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.2519, 1.3442, 1.2519], angles=[121.14, 121.14], turns=[1, -1]))

# ---------------------------------------------------------------------
# N5 family
# ---------------------------------------------------------------------
add(id="N5+_Vchain", formula="N5", charge=1, mult=1, point_group="C2v",
    source="[B] p.24 (CCSD(T)/6-311+G(2d))", method="CCSD(T)/6-311+G(2d)",
    build="chain2d",
    params=dict(bonds=[1.12, 1.33, 1.33, 1.12], angles=[166.4, 108.3, 166.4], turns=[1, 1, -1]))

add(id="N5-_bipyramid", formula="N5", charge=-1, mult=1, point_group="D3h",
    source="[B] p.26 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="trigonal_bipyramid_nocenter", params=dict(bond=1.4716, angle_apex=78.60))

add(id="N5-_pentagon", formula="N5", charge=-1, mult=1, point_group="D5h",
    source="[B] p.28 (CCSD(T)/aug-cc-pVTZ) - pentazolate", method="CCSD(T)/aug-cc-pVTZ",
    build="ring_regular", params=dict(n=5, edge=1.3294))

add(id="N5H_pentazole", formula="N5H", charge=0, mult=1, point_group="C2v",
    source="[GS] Fig.5 p.7129 (MP2(fc)/6-31G*) - pentazole 2",
    method="MP2(fc)/6-31G*",
    build="ring_with_H", params=dict(
        n=5, bonds=[1.327, 1.329, 1.348, 1.329, 1.327], angles=[114.1, 103.6, 103.6, 114.1],
        h_index=0, nh_bond=1.00),
    notes="Cycle N5 + H sur l'azote pyrrolique; longueur N-H estimee (non publiee).")

# ---------------------------------------------------------------------
# N6 family
# ---------------------------------------------------------------------
add(id="N6+_C2h_planar", formula="N6", charge=1, mult=2, point_group="C2h",
    source="[B] p.30 (B3LYP/aug-cc-pVDZ, 2Bg, planar)", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1233, 1.2999, 1.3427, 1.2999, 1.1233],
                angles=[167.30, 109.55, 109.55, 167.30], turns=[1, -1, 1, -1]))

add(id="N6+_C2v", formula="N6", charge=1, mult=2, point_group="C2v",
    source="[B] p.31 (B3LYP/aug-cc-pVDZ, 2Bg)", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1268, 1.3064, 1.3288, 1.3064, 1.1268],
                angles=[166.05, 119.32, 119.32, 166.05], turns=[1, -1, 1, -1]))

add(id="N6+_C2h_nonplanar", formula="N6", charge=1, mult=2, point_group="C2h",
    source="[B] p.30 (B3LYP/aug-cc-pVDZ, 2Ag, non-planaire)",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=6, shape="chain"),
    notes="Isomere non planaire (repliement hors-plan visible sur la figure); "
          "dièdres non publies -> relaxation GFN2-xTB.")

add(id="N6_D3h_prism", formula="N6", charge=0, mult=1, point_group="D3h",
    source="[B] p.31 (B3LYP/aug-cc-pVDZ) - prisme trigonal", method="B3LYP/aug-cc-pVDZ",
    build="prism", params=dict(n=3, edge_ring=1.5245, edge_vertical=1.4840))

add(id="N6_C2_book", formula="N6", charge=0, mult=1, point_group="C2",
    source="[B] p.33 (B3LYP/aug-cc-pVDZ) - 'book'", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=6, shape="ring6_fold"),
    notes="Cycle a 6 centres replie en 'livre'; angle de pliage non publie -> "
          "depart = hexagone plisse, relaxation GFN2-xTB.")

add(id="N6_D2_twisted", formula="N6", charge=0, mult=1, point_group="D2",
    source="[B] p.34 (B3LYP/aug-cc-pVDZ) - anneau torsade", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=6, shape="ring6_twist"),
    notes="Cycle a 6 centres en bateau torsade (D2); dièdre non publie -> "
          "depart = hexagone plisse alterne, relaxation GFN2-xTB.")

add(id="N6_C2h_chain", formula="N6", charge=0, mult=1, point_group="C2h",
    source="[B] p.34 (CCSD(T)/cc-pVDZ) - chaine plane", method="CCSD(T)/cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1521, 1.2614, 1.4672, 1.2614, 1.1521],
                angles=[171.54, 109.15, 109.15, 171.54], turns=[1, -1, 1, -1]))

add(id="N6-_C2_linked_triangles", formula="N6", charge=-1, mult=2, point_group="C2",
    source="[B] p.36 (B3LYP/aug-cc-pVDZ) - triangles lies", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=6, shape="two_triangles_shared_vertex"),
    notes="Deux cycles a 3 centres relies (structure 'linked triangles'); "
          "geometrie exacte 3D non publiee -> relaxation GFN2-xTB.")

add(id="N6-_Cs_chain", formula="N6", charge=-1, mult=2, point_group="Cs",
    source="[B] p.36 (B3LYP/aug-cc-pVDZ) - chaine en W", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1635, 1.2231, 1.4405, 1.3453, 1.2086],
                angles=[174.10, 114.20, 105.04, 126.03], turns=[1, -1, 1, -1]))

# ---------------------------------------------------------------------
# N7 family
# ---------------------------------------------------------------------
add(id="N7+_chain", formula="N7", charge=1, mult=1, point_group="C2v",
    source="[B] p.37 (B3LYP/aug-cc-pVDZ) - chaine en W", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1168, 1.3009, 1.3228, 1.3228, 1.3009, 1.1168],
                angles=[165.28, 109.58, 108.72, 108.72, 109.58],
                turns=[1, -1, 1, -1, 1]))

add(id="N7_C2v_logcarrier", formula="N7", charge=0, mult=2, point_group="C2v",
    source="[B] p.38 (B3LYP/aug-cc-pVDZ) - 'log carrier'", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=7, shape="cage", mult=2),
    notes="Cage bicyclique complexe ('log carrier'); dièdres non publies -> "
          "depart generique compact, relaxation GFN2-xTB (doublet).")

add(id="N7_C2_linearZZ", formula="N7", charge=0, mult=2, point_group="C2",
    source="[B] p.39 (B3LYP/aug-cc-pVDZ) - chaine ZZ", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1439, 1.2676, 1.3608, 1.3608, 1.2676, 1.1439],
                angles=[167.25, 122.09, 120.36, 120.36, 122.09],
                turns=[1, -1, 1, -1, 1]))

add(id="N7_Cs_ring_chain", formula="N7", charge=0, mult=2, point_group="Cs",
    source="[B] p.40 (B3LYP/aug-cc-pVDZ) - cycle a 5 + chaine",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=7, shape="ring5chain", mult=2),
    notes="Cycle a 5 centres + chaine de 2 atomes exocyclique; assignation "
          "angle cycle/exocyclique ambigue sur la figure -> relaxation GFN2-xTB.")

add(id="N7-_chain", formula="N7", charge=-1, mult=1, point_group="C2v",
    source="[B] p.41 (B3LYP/aug-cc-pVDZ) - chaine en W", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1694, 1.2228, 1.4149, 1.4149, 1.2228, 1.1694],
                angles=[174.46, 115.78, 102.70, 115.78, 174.46],
                turns=[1, -1, 1, -1, 1]))

# ---------------------------------------------------------------------
# N8 family
# ---------------------------------------------------------------------
add(id="N8_Oh_cube", formula="N8", charge=0, mult=1, point_group="Oh",
    source="[B] p.42 (B3LYP/aug-cc-pVDZ) - cubane N8", method="B3LYP/aug-cc-pVDZ",
    build="cube", params=dict(edge=1.5218))

add(id="N8_D2h_ring_pendant", formula="N8", charge=0, mult=1, point_group="D2h",
    source="[B] p.43 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=8, shape="ring6+2pendant"),
    notes="Cycle a 6 centres + 2 substituants exocycliques; depart generique, "
          "relaxation GFN2-xTB.")

add(id="N8_C2h_ladder", formula="N8", charge=0, mult=1, point_group="C2h",
    source="[B] p.44 (B3LYP/aug-cc-pVDZ) - 'ladder'", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=8, shape="ladder"),
    notes="Structure en 'echelle' (deux cycles a 4 fusionnes); dièdres non "
          "publies -> relaxation GFN2-xTB.")

add(id="N8_C2v_ring", formula="N8", charge=0, mult=1, point_group="C2v",
    source="[B] p.45 (B3LYP/aug-cc-pVDZ) - cycle a 8", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=8, shape="ring8"),
    notes="Cycle a 8 centres a liaisons alternees (~1.43/~1.24 A); les angles "
          "publies (~117.5 deg partout) ne closent pas un octogone plan "
          "(cycle probablement non-plan, type 'tub') -> relaxation GFN2-xTB.")

add(id="N8_C2v_branched", formula="N8", charge=0, mult=1, point_group="C2v",
    source="[B] p.46 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=8, shape="branched"),
    notes="Squelette ramifie avec centre tetracoordine (1.5713/1.1553/96.66); "
          "relaxation GFN2-xTB pour fixer la geometrie 3D.")

add(id="N8_C1_ZZE", formula="N8", charge=0, mult=1, point_group="C1",
    source="[B] p.47 (B3LYP/aug-cc-pVDZ) - chaine ZZE", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1361, 1.2614, 1.4152, 1.2428, 1.4152, 1.2554, 1.1334],
                angles=[168.41, 118.26, 123.14, 116.12, 109.95, 172.62],
                turns=[1, -1, 1, -1, 1, -1]))

add(id="N8_C2h_ZEZ", formula="N8", charge=0, mult=1, point_group="C2h",
    source="[B] p.48 (B3LYP/aug-cc-pVDZ) - chaine ZEZ", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1326, 1.2720, 1.3893, 1.2546, 1.3893, 1.2720, 1.1326],
                angles=[170.42, 114.89, 115.47, 115.47, 114.89, 170.42],
                turns=[1, -1, 1, 1, -1, 1]))

add(id="N8_Cs_ZEE", formula="N8", charge=0, mult=1, point_group="Cs",
    source="[B] p.49 (B3LYP/aug-cc-pVDZ) - chaine mixte",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=8, shape="chain"),
    notes="Chaine plane a 8 centres ; une valeur (longueur/angle) manquante "
          "dans la transcription des 7 liaisons -> relaxation GFN2-xTB.")

add(id="N8_D2h_pentalene", formula="N8", charge=0, mult=1, point_group="D2h",
    source="[GS] Fig.3 p.7128 (Becke3LYP/6-31G*) - octaazapentalene (12)",
    method="B3LYP/6-31G* (topologie exacte, geometrie relaxee GFN2-xTB)",
    build="xtb_generic", params=dict(n_atoms=8, shape="pentalene"),
    notes="Bicycle 5-5 fusionne (aromatique, 10 pi e-); depart = 2 pentagones "
          "fusionnes, relaxation GFN2-xTB (les longueurs de liaison publiees "
          "1.30-1.35 A sont utilisees comme depart).")

add(id="N8_C2h_EEE", formula="N8", charge=0, mult=1, point_group="C2h",
    source="[B] p.51 (B3LYP/aug-cc-pVDZ) - chaine EEE", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1344, 1.2590, 1.3965, 1.2505, 1.3965, 1.2590, 1.1344],
                angles=[170.38, 109.96, 108.23, 108.23, 109.96, 170.38],
                turns=[1, -1, 1, -1, 1, -1]))

add(id="N8_C2v_EZE", formula="N8", charge=0, mult=1, point_group="C2v",
    source="[B] p.52 (B3LYP/aug-cc-pVDZ) - chaine EZE", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1343, 1.2562, 1.4156, 1.2415, 1.4156, 1.2562, 1.1343],
                angles=[171.47, 109.86, 115.44, 115.44, 109.86, 171.47],
                turns=[1, -1, 1, 1, -1, 1]))

add(id="N8_Cs_pentagonal", formula="N8", charge=0, mult=1, point_group="Cs",
    source="[B] p.53 (B3LYP/aug-cc-pVDZ) - cycle a 5 + chaine",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=8, shape="ring5chain"),
    notes="Cycle a 5 centres + chaine de 3 atomes exocyclique; relaxation GFN2-xTB.")

add(id="N8-_C2h_chain", formula="N8", charge=-1, mult=2, point_group="C2h",
    source="[B] p.54 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1632, 1.2346, 1.3948, 1.3443, 1.3948, 1.2346, 1.1632],
                angles=[172.73, 115.31, 105.61, 105.61, 115.31, 172.73],
                turns=[1, -1, 1, 1, -1, 1]))

add(id="N8-_C2h_ZEZ", formula="N8", charge=-1, mult=2, point_group="C2h",
    source="[B] p.55 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1614, 1.2568, 1.3807, 1.3326, 1.3807, 1.2568, 1.1614],
                angles=[168.78, 111.44, 119.54, 119.54, 111.44, 168.78],
                turns=[1, -1, 1, 1, -1, 1]))

# ---------------------------------------------------------------------
# N9 family
# ---------------------------------------------------------------------
add(id="N9+_fused_rings", formula="N9", charge=1, mult=1, point_group="C2v",
    source="[B] p.56 (B3LYP/aug-cc-pVDZ) - cycles fusionnes 5,6",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=9, shape="fused56"),
    notes="Bicycle 5,6 fusionne (annulation); relaxation GFN2-xTB.")

add(id="N9+_chain", formula="N9", charge=1, mult=1, point_group="C2v",
    source="[B] p.57 (B3LYP/aug-cc-pVDZ) - chaine", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1170, 1.2985, 1.3174, 1.3281, 1.3281, 1.3174, 1.2985, 1.1170],
                angles=[164.92, 109.83, 108.23, 105.15, 105.15, 108.23, 109.83],
                turns=[1, -1, 1, -1, 1, -1, 1]))

add(id="N9_fused_rings", formula="N9", charge=0, mult=2, point_group="C2v",
    source="[B] p.58 (B3LYP/aug-cc-pVDZ) - cycles fusionnes",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=9, shape="fused56", mult=2),
    notes="Bicycle fusionne (doublet); relaxation GFN2-xTB.")

add(id="N9_chain", formula="N9", charge=0, mult=2, point_group="C2v",
    source="[B] p.59 (B3LYP/aug-cc-pVDZ) - chaine EEEE", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1355, 1.2632, 1.3682, 1.3058, 1.3058, 1.3682, 1.2632, 1.1355],
                angles=[169.80, 111.33, 106.97, 108.28, 106.97, 111.33, 169.80],
                turns=[1, -1, 1, -1, 1, -1, 1]))

add(id="N9-_chain_twistedW", formula="N9", charge=-1, mult=1, point_group="C2",
    source="[B] p.60 (B3LYP/aug-cc-pVDZ) - 'twisted W'", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1536, 1.2403, 1.4371, 1.2973, 1.2973, 1.4371, 1.2403, 1.1536],
                angles=[172.72, 113.90, 111.07, 118.77, 111.07, 113.90, 172.72],
                turns=[1, -1, 1, -1, 1, -1, 1]),
    notes="Denomme 'twisted W' dans la reference ; construit ici comme chaine "
          "plane symetrique (approximation).")

add(id="N9-_chain", formula="N9", charge=-1, mult=1, point_group="C2v",
    source="[B] p.61 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1572, 1.2285, 1.4400, 1.2933, 1.2933, 1.4400, 1.2285, 1.1572],
                angles=[173.55, 113.24, 105.89, 113.12, 105.89, 113.24, 173.55],
                turns=[1, -1, 1, -1, 1, -1, 1]))

add(id="N9-_ring_chain", formula="N9", charge=-1, mult=1, point_group="Cs",
    source="[B] p.62 (B3LYP/aug-cc-pVDZ) - cycle a 6 + chaine",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=9, shape="ring6chain"),
    notes="Cycle a 6 centres + chaine de 3 atomes exocyclique; la fermeture "
          "exacte du cycle a partir des valeurs publiees restait incoherente "
          "(erreur de fermeture > 0.5 A) -> relaxation GFN2-xTB.")

# ---------------------------------------------------------------------
# N10 family
# ---------------------------------------------------------------------
add(id="N10+_rings", formula="N10", charge=1, mult=2, point_group="C1",
    source="[B] p.63 (B3LYP/aug-cc-pVDZ)", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=10, shape="cage", mult=2),
    notes="Symetrie C1 (aucune contrainte) ; relaxation GFN2-xTB (doublet).")

add(id="N10+_chain", formula="N10", charge=1, mult=2, point_group="C1",
    source="[B] p.64 (B3LYP/aug-cc-pVDZ) - chaine", method="B3LYP/aug-cc-pVDZ",
    build="chain2d",
    params=dict(bonds=[1.1210, 1.3019, 1.3295, 1.2571, 1.3325, 1.2571, 1.3295, 1.3019, 1.1210],
                angles=[166.64, 109.63, 112.65, 116.35, 116.35, 112.65, 109.63, 166.64],
                turns=[1, -1, 1, -1, 1, -1, 1, -1]),
    notes="Chaine C1 approximee par une chaine plane symetrique (les valeurs "
          "gauche/droite legerement asymetriques dans la reference ont ete "
          "moyennees).")

add(id="N10_cage_C2v", formula="N10", charge=0, mult=1, point_group="C2v",
    source="[B] p.65 (B3LYP/aug-cc-pVDZ) - 'log carrier'", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=10, shape="cage"),
    notes="Cage bicyclique ('log carrier'); relaxation GFN2-xTB.")

add(id="N10_D5h_prism", formula="N10", charge=0, mult=1, point_group="D5h",
    source="[B] p.66 (B3LYP/aug-cc-pVDZ) - prisme pentagonal", method="B3LYP/aug-cc-pVDZ",
    build="prism", params=dict(n=5, edge_ring=1.4984, edge_vertical=1.5207))

add(id="N10_C3_cap", formula="N10", charge=0, mult=1, point_group="C3",
    source="[B] p.67 (B3LYP/aug-cc-pVDZ) - 'cap'", method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=10, shape="cage"),
    notes="Cage 'cap' (calotte); relaxation GFN2-xTB.")

add(id="N10_Cs_ring_chain", formula="N10", charge=0, mult=1, point_group="Cs",
    source="[B] p.68 (B3LYP/aug-cc-pVDZ) - cycle a 5/6 + chaine",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=10, shape="ring6chain"),
    notes="Cycle fusionne a une chaine de 4 atomes; nombre exact d'atomes du "
          "cycle incertain dans la transcription -> relaxation GFN2-xTB.")

add(id="N10_D2d_linked5rings", formula="N10", charge=0, mult=1, point_group="D2d",
    source="[B] p.69 + [GS] Fig.6 (21, D2d) - bispentazole",
    method="B3LYP/6-31G*",
    build="two_rings_ortho",
    params=dict(connect_len=1.377,
                ring=dict(bonds=[1.353, 1.344, 1.344, 1.353, 1.282],
                          angles=[112.7, 104.1, 104.1, 112.7])))

add(id="N10-_D2h_perp_rings", formula="N10", charge=-1, mult=2, point_group="D2h",
    source="[B] p.70 (B3LYP/aug-cc-pVDZ) - deux cycles perpendiculaires",
    method="B3LYP/aug-cc-pVDZ (topologie)",
    build="xtb_generic", params=dict(n_atoms=10, shape="linked_5rings", mult=2),
    notes="Deux cycles a 5 centres relies par une liaison simple et "
          "perpendiculaires entre eux (D2h) ; la fermeture exacte de "
          "chaque cycle a partir des valeurs publiees restait incoherente "
          "-> depart = deux pentagones coplanaires relies, relaxation "
          "GFN2-xTB (qui retablit la perpendicularite).")

# ---------------------------------------------------------------------
# Structures supplementaires de [GS] (poly-N.pdf) non dupliquees ci-dessus
# ---------------------------------------------------------------------
add(id="N8_Oh_cubane_GS", formula="N8", charge=0, mult=1, point_group="Oh",
    source="[GS] Fig.3 (11, Oh) - octaazacubane", method="Becke3LYP/6-31G*",
    build="cube", params=dict(edge=1.521))

add(id="N8_D2d_octaazacyclooctatetraene", formula="N8", charge=0, mult=1, point_group="D2d",
    source="[GS] Fig.3 (13, D2d)", method="Becke3LYP/6-31G* (topologie)",
    build="xtb_generic", params=dict(n_atoms=8, shape="ring8"),
    notes="Cycle a 8 centres 'tub' non-plan (cyclooctatetraene-like), liaisons "
          "alternees ~1.43/~1.24 A; dièdres de plissement non publies -> "
          "relaxation GFN2-xTB.")

add(id="N8_Cs_azidopentazole", formula="N8", charge=0, mult=1, point_group="Cs",
    source="[GS] Fig.3 (14, Cs) - azidopentazole", method="Becke3LYP/6-31G* (topologie)",
    build="xtb_generic", params=dict(n_atoms=8, shape="ring5chain"),
    notes="Cycle pentazole + groupe azoture -N3 exocyclique (3 atomes); "
          "relaxation GFN2-xTB (une longueur de liaison du raccord n'etait "
          "pas lisible avec certitude sur la figure).")

add(id="N10_D2d_bispentazole_GS", formula="N10", charge=0, mult=1, point_group="D2d",
    source="[GS] Fig.6 (21, D2d)", method="Becke3LYP/6-31G*",
    build="two_rings_ortho",
    params=dict(connect_len=1.377,
                ring=dict(bonds=[1.353, 1.344, 1.344, 1.353, 1.282],
                          angles=[112.7, 104.1, 104.1, 112.7])))

add(id="N10_D10h_ring", formula="N10", charge=0, mult=1, point_group="D10h",
    source="[GS] Fig.6 (23, D10h)", method="Becke3LYP/6-31G*",
    build="ring_regular", params=dict(n=10, edge=1.250))

add(id="N12_C2h_dipentazolyldiazene", formula="N12", charge=0, mult=1, point_group="C2h",
    source="[GS] Fig.7 (26a, C2h)", method="Becke3LYP/6-31G* (topologie)",
    build="xtb_generic", params=dict(n_atoms=12, shape="two_rings_plus_NN"),
    notes="Deux cycles pentazolyle relies par un pont N=N trans; relaxation GFN2-xTB.")

add(id="N20_Ih_dodecahedrane", formula="N20", charge=0, mult=1, point_group="Ih",
    source="[GS] Fig.7 (27, Ih)", method="Becke3LYP/6-31G*",
    build="dodecahedron", params=dict(edge=1.493))
