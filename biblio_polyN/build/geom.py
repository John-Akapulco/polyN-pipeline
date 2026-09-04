"""Primitives geometriques pour reconstruire des structures N_n a partir des
parametres (longueurs de liaison, angles) publies dans la bibliographie.

Toutes les longueurs sont en angstroms, les angles fournis en degres.
"""
import numpy as np


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def chain2d(bonds, angles, turns=None):
    """Chaine plane (zig-zag) dans le plan xy (z=0).

    bonds : liste de N-1 longueurs de liaison (atome i -> i+1)
    angles: liste de N-2 angles de valence (au sommet i+1, entre liaisons i et i+1)
    turns : liste de N-2 signes (+1/-1) donnant le sens du virage (zig-zag si
            alternes, meme signe -> polygone). Par defaut alternance stricte
            en commencant par +1.
    """
    n = len(bonds) + 1
    if turns is None:
        turns = [(-1) ** k for k in range(n - 2)]
    coords = np.zeros((n, 3))
    coords[1] = [bonds[0], 0.0, 0.0]
    heading = 0.0
    for i in range(1, n - 1):
        ext = 180.0 - angles[i - 1]
        heading += turns[i - 1] * np.deg2rad(ext)
        direction = np.array([np.cos(heading), np.sin(heading), 0.0])
        coords[i + 1] = coords[i] + bonds[i] * direction
    return coords


def regular_ngon(n, edge):
    """Polygone regulier plan (anneau D_nh), rayon derive de la longueur d'arete."""
    R = edge / (2 * np.sin(np.pi / n))
    pts = []
    for k in range(n):
        th = 2 * np.pi * k / n
        pts.append([R * np.cos(th), R * np.sin(th), 0.0])
    return np.array(pts)


def general_ring(bonds, angles):
    """Anneau ferme (n atomes), bonds/angles listes de longueur n (cycliques).
    Construit par propagation (meme sens de virage) ; renvoie (coords, closure_error).
    Si un seul angle manque (n-1 fournis), il est complete via la regle de
    somme des angles interieurs d'un polygone plan ((n-2)*180 deg).
    """
    n = len(bonds)
    angles = list(angles)
    if len(angles) == n - 1:
        missing = (n - 2) * 180.0 - sum(angles)
        angles = angles + [missing]
    coords = np.zeros((n, 3))
    coords[1] = [bonds[0], 0.0, 0.0]
    heading = 0.0
    for i in range(1, n - 1):
        ext = 180.0 - angles[i - 1]
        heading += np.deg2rad(ext)
        direction = np.array([np.cos(heading), np.sin(heading), 0.0])
        coords[i + 1] = coords[i] + bonds[i] * direction
    # fermeture : distance coords[-1]-coords[0] doit valoir bonds[-1]
    closure_error = np.linalg.norm(coords[-1] - coords[0]) - bonds[-1]
    return coords, closure_error


def tetrahedron(edge):
    base = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float)
    cur = 2 * np.sqrt(2)
    return base * (edge / cur)


def cube(edge):
    pts = []
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                pts.append([x, y, z])
    pts = np.array(pts, dtype=float)
    return pts * (edge / 2.0)


def prism_dnh(n, edge_ring, edge_vertical):
    """Prisme droit eclipse D_nh : deux n-gones paralleles superposes."""
    R = edge_ring / (2 * np.sin(np.pi / n))
    h = edge_vertical
    top, bot = [], []
    for k in range(n):
        th = 2 * np.pi * k / n
        top.append([R * np.cos(th), R * np.sin(th), h / 2])
        bot.append([R * np.cos(th), R * np.sin(th), -h / 2])
    return np.array(top + bot)


def trigonal_bipyramid_nocenter(bond_ax_eq, angle_eq_ax_eq_deg, angle_ax_eq_ax_deg=None):
    """N5 D3h sans atome central : 2 sommets axiaux + 3 sommets equatoriaux,
    chaque axial lie aux 3 equatoriaux (6 liaisons), pas de liaison eq-eq ni ax-ax.
    """
    L = bond_ax_eq
    c1 = np.cos(np.deg2rad(angle_eq_ax_eq_deg))  # angle eq-Nax-eq (au sommet axial)
    # h^2 - r^2/2 = c1*L^2 ;  r^2+h^2 = L^2
    # => h^2(1) ... resoudre :
    # r^2 = L^2 - h^2
    # h^2 - (L^2-h^2)/2 = c1 L^2  => h^2*(3/2) = c1 L^2 + L^2/2 => h^2 = L^2*(c1+0.5)*(2/3)
    h2 = L ** 2 * (c1 + 0.5) * (2.0 / 3.0)
    h = np.sqrt(max(h2, 1e-8))
    r2 = L ** 2 - h2
    r = np.sqrt(max(r2, 1e-8))
    apex = [np.array([0, 0, h]), np.array([0, 0, -h])]
    eq = []
    for k in range(3):
        th = 2 * np.pi * k / 3
        eq.append(np.array([r * np.cos(th), r * np.sin(th), 0.0]))
    return np.array(apex + eq)


def puckered_4ring_d2d(edge, angle_deg):
    """N4 D2d anneau plisse (cyclobutane-like), symetrie S4.
    Parametrise par la longueur d'arete et l'angle N-N-N ; le pliage (dihedre)
    est resolu numeriquement pour etre coherent avec S4 (paire d'atomes hauts,
    paire d'atomes bas, alternees)."""
    from scipy.optimize import brentq
    L = edge
    theta = np.deg2rad(angle_deg)

    def build(phi):
        # 4 atomes sur un anneau S4 : (R cosA, R sinA, +z),(R cosB,R sinB,-z) alternes
        # parametrise par R, z relies par phi (angle de pliage) ; on exprime via
        # positions sur un anneau de rayon R a des angles 0,90,180,270 avec z alterne +-z
        pass

    # Construction directe : sommets a angles 0,90,180,270 sur un cercle de rayon R,
    # z alterne +d,-d,+d,-d. Par symetrie D2d toutes les aretes et tous les angles
    # N-N-N sont egaux automatiquement pour un choix (R,d). On resout (R,d) pour
    # matcher (edge, angle).
    def geom(R, d):
        pts = []
        for k in range(4):
            a = np.pi / 2 * k
            z = d if k % 2 == 0 else -d
            pts.append([R * np.cos(a), R * np.sin(a), z])
        return np.array(pts)

    def residual(R, d):
        pts = geom(R, d)
        b = np.linalg.norm(pts[1] - pts[0])
        v1 = pts[0] - pts[1]
        v2 = pts[2] - pts[1]
        ang = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        return b, ang

    # recherche 1D : pour chaque d, ajuste R pour matcher edge exactement, puis
    # cherche d qui matche l'angle
    def edge_ok_R(d):
        # bissection sur R pour que la distance 0-1 = L
        lo, hi = 1e-6, 5 * L
        def f(R):
            pts = geom(R, d)
            return np.linalg.norm(pts[1] - pts[0]) - L
        return brentq(f, lo, hi)

    def ang_err(d):
        R = edge_ok_R(d)
        _, ang = residual(R, d)
        return ang - angle_deg

    d_lo, d_hi = 1e-6, L
    try:
        d = brentq(ang_err, d_lo, d_hi)
    except ValueError:
        d = L * 0.3
    R = edge_ok_R(d)
    return geom(R, d)


def rectangle(a, b):
    """4 sommets d'un rectangle a x b, plan xy, dans l'ordre du cycle."""
    return np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a, b, 0],
        [0, b, 0],
    ], dtype=float)


def dodecahedron(edge):
    """20 sommets d'un dodecaedre regulier (groupe Ih), longueur d'arete donnee."""
    phi = (1 + np.sqrt(5)) / 2
    pts = []
    for x in (-1, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                pts.append([x, y, z])
    for a in (-1, 1):
        for b in (-1, 1):
            pts.append([0, a / phi, b * phi])
            pts.append([a / phi, b * phi, 0])
            pts.append([b * phi, 0, a / phi])
    pts = np.array(pts, dtype=float)
    # arete actuelle = 2/phi
    cur_edge = 2.0 / phi
    return pts * (edge / cur_edge)


def ring_plus_chain(ring_bonds, ring_angles, attach_atom, exo_angle, chain_bonds,
                     chain_angles=None):
    """Cycle plan (n atomes, n bonds/angles cycliques, fermeture approximative
    si les valeurs publiees ne sont pas parfaitement coherentes) avec une
    chaine plane greffee sur l'atome `attach_atom` (index 0-based), orientee
    vers l'exterieur du cycle. `exo_angle` = angle (deg) entre la liaison du
    cycle entrante et la premiere liaison de la chaine. `chain_angles`
    (optionnel, longueur len(chain_bonds)-1) donne les angles suivants le
    long de la chaine (par defaut 170 deg, quasi lineaire, alternance)."""
    ring, closure_err = general_ring(ring_bonds, ring_angles)
    n = len(ring)
    center = ring.mean(axis=0)
    a = ring[attach_atom]
    prev_atom = ring[(attach_atom - 1) % n]
    outward = unit(a - center)
    ref = unit(a - prev_atom)
    heading0 = np.arctan2(ref[1], ref[0])
    best = None
    for sign in (+1, -1):
        heading = heading0 + sign * np.deg2rad(180 - exo_angle)
        d = np.array([np.cos(heading), np.sin(heading), 0.0])
        test = a + chain_bonds[0] * d
        score = np.dot(test - center, outward)
        if best is None or score > best[0]:
            best = (score, sign)
    chosen = best[1]
    if chain_angles is None:
        chain_angles = [170.0] * (len(chain_bonds) - 1)
    coords = [a]
    heading = heading0 + chosen * np.deg2rad(180 - exo_angle)
    pos = a + chain_bonds[0] * np.array([np.cos(heading), np.sin(heading), 0.0])
    coords.append(pos)
    turn = chosen
    for i in range(1, len(chain_bonds)):
        ext = 180.0 - chain_angles[i - 1]
        turn = -turn
        heading += turn * np.deg2rad(ext)
        direction = np.array([np.cos(heading), np.sin(heading), 0.0])
        pos = pos + chain_bonds[i] * direction
        coords.append(pos)
    chain_pts = np.array(coords[1:])
    return ring, chain_pts, closure_err


def ring_with_h(n, bonds, angles, h_index, nh_bond):
    """Cycle plan de n atomes + un H exocyclique porte par l'atome h_index,
    dirige selon la bissectrice exterieure de cet atome."""
    ring, closure_err = general_ring(bonds, angles)
    center = ring.mean(axis=0)
    a = ring[h_index]
    outward = unit(a - center)
    h_pos = a + nh_bond * outward
    return ring, h_pos, closure_err


def _to_c(pts2d):
    return pts2d[:, 0] + 1j * pts2d[:, 1]


def fused_bicyclic(n1, n2, edge_len=1.35):
    """Deux cycles (n1 et n2 atomes) partageant une arete commune, tous deux
    dans le plan xy (depart plan, ex: pentalene, azulene-like)."""
    ring1 = regular_ngon(n1, edge_len)
    ring2 = regular_ngon(n2, edge_len)
    z1 = _to_c(ring1[:, :2])
    z2 = _to_c(ring2[:, :2])
    src0, src1 = z2[0], z2[1]
    dst0, dst1 = z1[1], z1[0]  # inversion pour placer ring2 de l'autre cote
    scale_rot = (dst1 - dst0) / (src1 - src0)
    z2t = (z2 - src0) * scale_rot + dst0
    ring2_t = np.stack([z2t.real, z2t.imag, np.zeros_like(z2t.real)], axis=1)
    combined = np.vstack([ring1, ring2_t[2:]])
    return combined


def two_rings_plus_bridge(n_ring, edge_ring, bridge_len, n_bridge_atoms):
    ringA = regular_ngon(n_ring, edge_ring)
    ringB = regular_ngon(n_ring, edge_ring)
    RA = edge_ring / (2 * np.sin(np.pi / n_ring))
    sep = 2 * RA + bridge_len * (n_bridge_atoms + 1)
    ringB = ringB + np.array([sep, 0, 0])
    bridge = []
    attachA = ringA[0]
    attachB = ringB[0]
    for i in range(n_bridge_atoms):
        frac = (i + 1) / (n_bridge_atoms + 1)
        bridge.append(attachA + frac * (attachB - attachA))
    if bridge:
        return np.vstack([ringA, ringB, np.array(bridge)])
    return np.vstack([ringA, ringB])


def two_triangles_linked(edge, connect_len):
    t1 = regular_ngon(3, edge)
    t2 = regular_ngon(3, edge)
    c1 = t1.mean(axis=0)
    c2 = t2.mean(axis=0)
    # eloigne les triangles le long de x, sommet 0 de chacun pointant vers l'autre
    t1 = t1 - t1[0]
    t2 = t2 - t2[0]
    t2 = -t2  # oriente vers -x
    t2[:, 0] += connect_len
    return np.vstack([t1, t2])


def ladder_2x4(bond=1.45):
    rows = []
    for i in range(4):
        rows.append([i * bond, 0.6 * bond, 0.0])
    for i in range(4):
        rows.append([i * bond, -0.6 * bond, 0.0])
    return np.array(rows)


def fibonacci_sphere(n, radius):
    pts = []
    ga = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(max(0.0, 1 - z * z))
        th = ga * i
        pts.append([r * np.cos(th) * radius, r * np.sin(th) * radius, z * radius])
    return np.array(pts)


def ring_plus_chain_guess(n_ring, n_chain, edge):
    ring = regular_ngon(n_ring, edge)
    center = ring.mean(axis=0)
    attach = ring[0]
    outward = unit(attach - center)
    # petite composante en z pour eviter la colinearite parfaite (aide xtb)
    perp = np.array([-outward[1], outward[0], 0.3])
    perp = unit(perp)
    chain = []
    pos = attach
    direction = outward
    for i in range(n_chain):
        pos = pos + edge * direction
        chain.append(pos.copy())
        direction = unit(0.6 * direction + 0.4 * perp)
    return np.vstack([ring, np.array(chain)])


def generic_guess(shape, n_atoms, edge=1.40):
    if shape.startswith("ring") and shape.endswith("chain") and shape[4].isdigit():
        n_ring = int(shape[4])
        n_chain = n_atoms - n_ring
        return ring_plus_chain_guess(n_ring, n_chain, edge)
    if shape.startswith("ring") and shape[4:].isdigit():
        n_ring = int(shape[4:])
        r = regular_ngon(n_ring, edge)
        r[:, 2] = [0.15 * edge * ((-1) ** k) for k in range(n_ring)]
        return r
    if shape == "ring3+pendant":
        tri = regular_ngon(3, edge)
        pend = tri[0] + np.array([edge * 0.6, 0.0, edge * 1.2])
        return np.vstack([tri, pend])
    if shape in ("ring6_fold", "ring6_twist"):
        hexr = regular_ngon(6, edge)
        d = 0.35 * edge
        if shape == "ring6_fold":
            zs = [d, d, d, -d, -d, -d]
        else:
            zs = [d, -d, d, -d, d, -d]
        hexr[:, 2] = zs
        return hexr
    if shape == "ring6+2pendant":
        hexr = regular_ngon(6, edge)
        center = hexr.mean(axis=0)
        p1 = hexr[0] + unit(hexr[0] - center) * edge
        p2 = hexr[3] + unit(hexr[3] - center) * edge
        return np.vstack([hexr, p1, p2])
    if shape == "two_triangles_shared_vertex":
        return two_triangles_linked(edge, connect_len=1.45)
    if shape == "ladder":
        return ladder_2x4(edge)
    if shape == "pentalene":
        return fused_bicyclic(5, 5, edge)
    if shape == "fused56":
        return fused_bicyclic(5, 6, edge)
    if shape == "two_rings_plus_NN":
        return two_rings_plus_bridge(5, edge, 1.30, 2)
    if shape == "linked_5rings":
        return two_rings_plus_bridge(5, edge, 1.35, 0)
    if shape == "chain":
        bonds = [edge] * (n_atoms - 1)
        angles = [112.0] * (n_atoms - 2)
        return chain2d(bonds, angles)
    if shape == "branched":
        return fibonacci_sphere(n_atoms, radius=2.1)
    if shape == "cage":
        radius = 0.72 * edge * np.sqrt(n_atoms)
        return fibonacci_sphere(n_atoms, radius=radius)
    # repli generique
    return fibonacci_sphere(n_atoms, radius=1.1 * edge * np.sqrt(n_atoms))


def two_rings_orthogonal(ring_a, ring_b, connect_len, attach_index=0):
    """Relie deux cycles plans (deja construits, listes Nx3 dans le plan xy,
    premier atome = point d'attache) par une liaison simple perpendiculaire :
    ring_a reste dans le plan xy, ring_b est place dans le plan xz, translate
    de sorte que la distance entre l'atome d'attache de A et celui de B vaille
    connect_len (les deux cycles ne se recouvrent pas)."""
    a = ring_a.copy()
    b = ring_b.copy()
    # centre A sur son atome d'attache a l'origine (deja le cas si attach_index0=0)
    a = a - a[attach_index]
    # oriente A vers les x negatifs (son premier voisin est vers +x par
    # construction de general_ring) pour eviter tout recouvrement avec B,
    # place lui vers les x positifs juste apres.
    a[:, 0] *= -1
    a[:, 1] *= -1
    # tourne B (dans son plan xy) pour le mettre dans le plan xz
    b = b - b[attach_index]
    b_rot = np.zeros_like(b)
    b_rot[:, 0] = b[:, 0]
    b_rot[:, 2] = b[:, 1]
    b_rot[:, 1] = 0.0
    # place B le long de +x a distance connect_len de l'atome d'attache de A
    b_rot[:, 0] += connect_len
    return a, b_rot
