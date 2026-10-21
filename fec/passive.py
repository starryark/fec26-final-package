"""MAST passive-structure currents from measured coil currents and Ip (inputs only).

Toroidally continuous MAST conductors (vacuum vessel, centre-column Inconel casing and rings,
coil cases, P2 support arms and divertor plates) obey, for each axisymmetric piece i,

    sum_j L_ij dI_j/dt + R_i I_i = -dPhi_i/dt,     Phi_i = 2*pi*psi_ext(piece i)  [Wb]

where psi_ext is the flux of the measured P2-P6 + solenoid currents and of the plasma current
(nominal distribution x measured Ip).  Two models:

  C1  full L/R circuit with mutual coupling, integrated per shot from before the solenoid
      pre-charge (all currents ~0) on a 0.5 ms grid; exact normal-mode integration of the
      generalised eigenproblem L v = tau R v for piecewise-linear drive.
  C5  quasi-steady I_i = -(dPhi_i/dt)/R_i (no inductive shielding), from the +-5 ms box means.

Sign convention as fec.machines: positive current flows in +phi (same direction as Ip),
psi = sum G*I in Wb/rad.  Geometry is the MAST EFIT passive machine description published in
FAIR-MAST level-2 `pf_passive` (source pfcoils.dat_M4 of UKAEA mast-geometry, shot range
0-40000; fetched for shot 30420, geometry arrays only).  Resistances are geometric
(rho * 2*pi*R / area) with handbook resistivities; see
docs/research/MAST_PASSIVE_MODEL_20261007.md for every number and its confidence.
No MAST equilibrium reconstruction, label or EFIT output is read anywhere.

CLI (writes only under FEC_WORK/passive):
    python -B fec/passive.py summary
    python -B fec/passive.py signals [--workers 6]
    python -B fec/passive.py frames --model c1 --rscale 1 [--psi]
    python -B fec/passive.py frames --model c5 --rscale 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fec.green import element_filaments, greens_psi  # noqa: E402
from fec.paths import WORK  # noqa: E402

MU0 = 4e-7 * np.pi
OUT = WORK / "passive"

# ----------------------------------------------------------------------------- machine data
# FAIR-MAST level 2, group pf_passive, variables <group>_{r,z,width,height,shapeAngle1,
# shapeAngle2} (coil_cases: r,z,width,height), shot 30420, s3.echo.stfc.ac.uk/mast/level2.
# Attributes: content "geometry of the passive structures for MAST", shotRangeStart 0,
# shotRangeStop 40000, source https://git.ccfe.ac.uk/MAST-U/mast-geometry/sources/pfcoils.dat_M4.
# Rows: (R, Z, dR, dZ, angle1, angle2) [m, m, m, m, deg, deg], EFIT parallelogram convention.
RAW_GEOMETRY = {
    "incon": [  # centre-column Inconel casing ("Centre column Inconel passive filaments")
        (0.177192, 1.797527, 0.014454, 0.154946, 0.0, 0.0),
        (0.173465, 1.505047, 0.007, 0.430013, 0.0, 0.0),
        (0.173465, 1.075034, 0.007, 0.430013, 0.0, 0.0),
        (0.173465, 0.64502, 0.007, 0.430013, 0.0, 0.0),
        (0.173465, 0.215007, 0.007, 0.430013, 0.0, 0.0),
        (0.173465, -0.215007, 0.007, 0.430013, 0.0, 0.0),
        (0.173465, -0.64502, 0.007, 0.430013, 0.0, 0.0),
        (0.173465, -1.075034, 0.007, 0.430013, 0.0, 0.0),
        (0.173465, -1.505047, 0.007, 0.430013, 0.0, 0.0),
        (0.177191, -1.797527, 0.014454, 0.154946, 0.0, 0.0),
    ],
    "ring": [  # centre-column rings ("Centre column ring passive filaments")
        (0.149301, 2.25, 0.031398, 0.081, 0.0, 0.0),
        (0.202025, 1.858692, 0.02405, 0.024059, 0.0, 0.0),
        (0.216475, 1.589722, 0.04705, 0.025, 0.0, 0.0),
        (0.216475, 1.364722, 0.04705, 0.025, 0.0, 0.0),
        (0.196031, 1.188222, 0.034938, 0.025, 0.0, 0.0),
        (0.196031, -1.188222, 0.034938, 0.025, 0.0, 0.0),
        (0.216475, -1.364722, 0.04705, 0.025, 0.0, 0.0),
        (0.216475, -1.589722, 0.04705, 0.025, 0.0, 0.0),
        (0.202025, -1.858692, 0.02405, 0.024059, 0.0, 0.0),
        (0.149301, -2.25, 0.031398, 0.081, 0.0, 0.0),
    ],
    "rodgr": [  # centre-rod armour layer (graphite tiles, R 0.177-0.195 m); excluded by default
        (0.228482, 1.654722, 0.103035, 0.105, 0.0, 0.0),
        (0.228482, 1.477222, 0.103035, 0.2, 0.0, 0.0),
        (0.227709, 1.276472, 0.101489, 0.1515, 0.0, 0.0),
        (0.196463, 1.129611, 0.038997, 0.092222, 0.0, 0.0),
        (0.186105, 0.812625, 0.01828, 0.54175, 0.0, 0.0),
        (0.186105, 0.270875, 0.01828, 0.54175, 0.0, 0.0),
        (0.186105, -0.270875, 0.01828, 0.54175, 0.0, 0.0),
        (0.186105, -0.812625, 0.01828, 0.54175, 0.0, 0.0),
        (0.196463, -1.129611, 0.038997, 0.092222, 0.0, 0.0),
        (0.227709, -1.276472, 0.101489, 0.1515, 0.0, 0.0),
        (0.228482, -1.477222, 0.103035, 0.2, 0.0, 0.0),
        (0.228482, -1.654722, 0.103035, 0.105, 0.0, 0.0),
    ],
    "botcol": [(0.235377, -2.025, 0.047142, 0.3, 0.0, 295.324341)],
    "topcol": [(0.235608, 2.025, 0.04668, 0.3, 0.0, 64.675667)],
    "endcrown_u": [(0.1625, 2.45825, 0.235, 0.0835, 0.0, 0.0)],
    "endcrown_l": [(0.1625, -2.45825, 0.235, 0.0835, 0.0, 0.0)],
    "vertw": [  # outer cylindrical vessel wall, 20 mm
        (2.00993, 1.82, 0.02, 0.52, 0.0, 0.0),
        (2.00993, 1.3, 0.02, 0.52, 0.0, 0.0),
        (2.00993, 0.78, 0.02, 0.52, 0.0, 0.0),
        (2.00993, 0.26, 0.02, 0.52, 0.0, 0.0),
        (2.00993, -0.26, 0.02, 0.52, 0.0, 0.0),
        (2.00993, -0.78, 0.02, 0.52, 0.0, 0.0),
        (2.00993, -1.3, 0.02, 0.52, 0.0, 0.0),
        (2.00993, -1.82, 0.02, 0.52, 0.0, 0.0),
    ],
    "uhorw": [  # upper horizontal vessel end plate, 20 mm, plus corner pieces
        (0.29018, 2.1975, 0.0795, 0.045, 0.0, 0.0),
        (0.53868, 2.21, 0.4175, 0.02, 0.0, 0.0),
        (0.95618, 2.21, 0.4175, 0.02, 0.0, 0.0),
        (1.37368, 2.21, 0.4175, 0.02, 0.0, 0.0),
        (1.79118, 2.21, 0.4175, 0.02, 0.0, 0.0),
        (2.042966, 2.15, 0.086072, 0.14, 0.0, 0.0),
    ],
    "lhorw": [
        (0.29018, -2.1975, 0.0795, 0.045, 0.0, 0.0),
        (0.53868, -2.21, 0.4175, 0.02, 0.0, 0.0),
        (0.95618, -2.21, 0.4175, 0.02, 0.0, 0.0),
        (1.37368, -2.21, 0.4175, 0.02, 0.0, 0.0),
        (1.79118, -2.21, 0.4175, 0.02, 0.0, 0.0),
        (2.042966, -2.15, 0.086072, 0.14, 0.0, 0.0),
    ],
    "mid": [  # box-section rings near the end plates (outer R~1.66, inner R~0.79)
        (1.655922, 1.8885, 0.135634, 0.007, 0.0, 0.0),
        (1.657975, 1.95, 0.006776, 0.116, 0.0, 0.0),
        (1.655922, 2.0101, 0.135634, 0.007, 0.0, 0.0),
        (1.655922, -1.8885, 0.135634, 0.007, 0.0, 0.0),
        (1.657975, -1.95, 0.006776, 0.116, 0.0, 0.0),
        (1.655922, -2.0101, 0.135634, 0.007, 0.0, 0.0),
        (0.789741, 1.8885, 0.134411, 0.007, 0.0, 0.0),
        (0.791631, 1.95, 0.006709, 0.116, 0.0, 0.0),
        (0.789741, 2.0101, 0.134411, 0.007, 0.0, 0.0),
        (0.789741, -1.8885, 0.134411, 0.007, 0.0, 0.0),
        (0.791631, -1.95, 0.006709, 0.116, 0.0, 0.0),
        (0.789741, -2.0101, 0.134411, 0.007, 0.0, 0.0),
    ],
    "p2uarm": [
        (0.35325, 1.630755, 0.1265, 0.041011, 315.0, 0.0),
        (0.50977, 1.562, 0.18654, 0.03, 0.0, 0.0),
        (0.68272, 1.650599, 0.04667, 0.147198, 0.0, 39.999992),
    ],
    "p2larm": [
        (0.35325, -1.630755, 0.1265, 0.041011, 45.0, 0.0),
        (0.50977, -1.562, 0.18654, 0.03, 0.0, 0.0),
        (0.68272, -1.650599, 0.04667, 0.147198, 0.0, 320.0),
    ],
    "p2udivpl": [
        (0.494087, 1.594916, 0.155173, 0.035833, 0.0, 0.0),
        (0.619821, 1.633718, 0.096295, 0.032635, 40.000023, 0.0),
    ],
    "p2ldivpl": [
        (0.492678, -1.594597, 0.157991, 0.035194, 0.0, 0.0),
        (0.619821, -1.633718, 0.096295, 0.032635, 319.999969, 0.0),
    ],
}
# coil_cases rows as published (R, Z, "width", "height"): the two size fields are swapped
# relative to the other groups (verified: e.g. the P4 inner case wall at R=1.406 must span
# Z 1.005-1.192 around the shipped P4 conductors, i.e. dZ=|width|, dR=|height|); the sign of
# "width" follows sign(Z).  Order: P2L, P2U, P3L (8), P3U (8), P4L, P4U, P5L, P5U, P6L, P6U.
RAW_COIL_CASES = [
    (0.436606, -1.7675, -0.289, 0.003), (0.498518, -1.9135, -0.003, 0.126824),
    (0.563431, -1.7705, -0.289, 0.003), (0.501518, -1.6245, -0.003, 0.126824),
    (0.436606, 1.7675, 0.289, 0.003), (0.498518, 1.9135, 0.003, 0.126824),
    (0.563431, 1.7705, 0.289, 0.003), (0.501518, 1.6245, 0.003, 0.126824),
    (1.041431, -1.1, -0.083206, 0.003), (1.053824, -1.152801, -0.022397, 0.004906),
    (1.09993, -1.1655, -0.003, 0.066376), (1.146037, -1.152801, -0.022397, 0.004906),
    (1.158431, -1.1, -0.083206, 0.003), (1.145892, -1.047199, -0.022397, 0.004614),
    (1.09993, -1.0345, -0.003, 0.066376), (1.054089, -1.047199, -0.022397, 0.004375),
    (1.041431, 1.1, 0.083206, 0.003), (1.053824, 1.152801, 0.022397, 0.004906),
    (1.09993, 1.1655, 0.003, 0.066376), (1.146037, 1.152801, 0.022397, 0.004906),
    (1.158431, 1.1, 0.083206, 0.003), (1.145892, 1.047199, 0.022397, 0.004614),
    (1.09993, 1.0345, 0.003, 0.066376), (1.054089, 1.047199, 0.022397, 0.004375),
    (1.40643, -1.0985, -0.187, 0.003), (1.49843, -1.1935, -0.003, 0.187),
    (1.593431, -1.1015, -0.187, 0.003), (1.50143, -1.0065, -0.003, 0.187),
    (1.40643, 1.0985, 0.187, 0.003), (1.49843, 1.1935, 0.003, 0.187),
    (1.593431, 1.1015, 0.187, 0.003), (1.50143, 1.0065, 0.003, 0.187),
    (1.556604, -0.4985, -0.187, 0.003), (1.649022, -0.5935, -0.003, 0.187836),
    (1.744441, -0.5015, -0.187, 0.003), (1.652022, -0.4065, -0.003, 0.187836),
    (1.556604, 0.4985, 0.187, 0.003), (1.649022, 0.5935, 0.003, 0.187836),
    (1.744441, 0.5015, 0.187, 0.003), (1.652022, 0.4065, 0.003, 0.187836),
    (1.409431, -0.8985, -0.152, 0.003), (1.440431, -0.976, -0.003, 0.065),
    (1.47443, -0.9015, -0.152, 0.003), (1.443431, -0.824, -0.003, 0.065),
    (1.409431, 0.8985, 0.152, 0.003), (1.440431, 0.976, 0.003, 0.065),
    (1.47443, 0.9015, 0.152, 0.003), (1.443431, 0.824, 0.003, 0.065),
]
CASE_NAMES = (["case_p2l"] * 4 + ["case_p2u"] * 4 + ["case_p3l"] * 8 + ["case_p3u"] * 8 +
              ["case_p4l"] * 4 + ["case_p4u"] * 4 + ["case_p5l"] * 4 + ["case_p5u"] * 4 +
              ["case_p6l"] * 4 + ["case_p6u"] * 4)

# Resistivities [ohm m] at room temperature (handbook values; see the model doc).
RHO_SS = 7.3e-7        # austenitic stainless steel 304/316 (7.2-7.4e-7)
RHO_INCONEL = 1.25e-6  # Inconel 718/625 (1.25-1.29e-6)
RHO_GRAPHITE = 1.0e-5  # only used if 'rodgr' is explicitly included (tiles: not continuous)
MATERIAL = {"incon": RHO_INCONEL, "rodgr": RHO_GRAPHITE}
DEFAULT_EXCLUDE = ("rodgr",)

# Reporting groups (Berkery 2021-like segment families, split upper/lower).
def report_group(name, z):
    ul = "u" if z > 0 else "l"
    if name.startswith("case_"):
        return name
    base = {"incon": "cc_casing", "ring": "cc_rings", "rodgr": "cc_armour",
            "botcol": "cc_end", "topcol": "cc_end", "endcrown_u": "cc_endcrown",
            "endcrown_l": "cc_endcrown", "vertw": "outer_wall", "uhorw": "end_plate",
            "lhorw": "end_plate", "mid": "mid_rings", "p2uarm": "p2_arm", "p2larm": "p2_arm",
            "p2udivpl": "p2_divplate", "p2ldivpl": "p2_divplate"}[name]
    return f"{base}_{ul}"


def raw_elements(exclude=DEFAULT_EXCLUDE):
    """[(group, R, Z, dR, dZ, a1, a2, rho)] for every published passive element."""
    out = []
    for g, rows in RAW_GEOMETRY.items():
        if g in exclude:
            continue
        for (R, Z, W, H, a1, a2) in rows:
            out.append((g, R, Z, abs(W), abs(H), a1, a2, MATERIAL.get(g, RHO_SS)))
    if "coil_cases" not in exclude:
        for name, (R, Z, w, h) in zip(CASE_NAMES, RAW_COIL_CASES):
            out.append((name, R, Z, abs(h), abs(w), 0.0, 0.0, RHO_SS))
    return out


def _split(R, Z, W, H, a1, a2, piece_len):
    """Split an EFIT parallelogram into nu x nv sub-parallelograms (same angles)."""
    nu = max(1, int(np.ceil(W / piece_len)))
    nv = max(1, int(np.ceil(H / piece_len)))
    out = []
    for iu in range(nu):
        for iv in range(nv):
            x = ((iu + 0.5) / nu - 0.5) * W
            z = ((iv + 0.5) / nv - 0.5) * H
            if a1 not in (0.0, 90.0):
                x = x + z / np.tan(np.deg2rad(a1))
            if a2 not in (0.0, 90.0):
                z = z + x * np.tan(np.deg2rad(a2))
            out.append((R + x, Z + z, W / nu, H / nv, a1, a2))
    return out


def _psi_points(pr, pz, fr, fz, fw, seg, nseg, chunk=400):
    """psi at points (pr,pz) per unit current of each segment: (Np, nseg).

    fr, fz: filament positions; fw: filament weight (fraction of its segment's current);
    seg: segment index of each filament.
    """
    out = np.zeros((pr.size, nseg))
    S = np.zeros((fr.size, nseg))
    S[np.arange(fr.size), seg] = fw
    for i in range(0, pr.size, chunk):
        g = greens_psi(pr[i:i + chunk, None], pz[i:i + chunk, None], fr[None, :], fz[None, :])
        out[i:i + chunk] = g @ S
    return out


def nominal_plasma(kind="ellipse", R0=0.90, Z0=0.0, a=0.55, kappa=1.8, h=0.015):
    """Filaments (R, Z, weight) of a nominal plasma current distribution (sum weight = 1)."""
    if kind == "filament":
        return np.array([R0]), np.array([Z0]), np.array([1.0])
    r = np.arange(R0 - a, R0 + a + 1e-9, h)
    z = np.arange(Z0 - kappa * a, Z0 + kappa * a + 1e-9, h)
    RR, ZZ = np.meshgrid(r, z)
    rho2 = ((RR - R0) / a) ** 2 + ((ZZ - Z0) / (kappa * a)) ** 2
    w = np.clip(1 - rho2, 0, None)
    m = w > 0
    return RR[m], ZZ[m], w[m] / w[m].sum()


class PassiveModel:
    """Axisymmetric MAST passive conductors as circuits (pieces), with coupling matrices.

    Attributes (Np = number of pieces):
      piece: list of (group, R, Z, dR, dZ, a1, a2); group per piece in `groups`
      Rres (Np,) ohm; L (Np,Np) H; Mpc (Np,C) Wb per A of each MAST_COILS input column
      (same per-column scaling as fec.machines.mast_setup); MpP (Np,) Wb per A of Ip;
      Gg (Ngrid, Np) psi [Wb/rad] on the 65x65 grid (row-major Z,R) per A in each piece.
    """

    def __init__(self, piece_len=0.07, exclude=DEFAULT_EXCLUDE, plasma="ellipse",
                 sol_fraction=None, cache=True):
        from fec.geometry import machine_geometry
        from fec.machines import MAST_COILS, MAST_SOL_TURN_FRACTION
        self.sol_fraction = MAST_SOL_TURN_FRACTION if sol_fraction is None else sol_fraction
        self.coils = list(MAST_COILS)
        self.piece_len, self.exclude, self.plasma = piece_len, tuple(exclude), plasma
        pieces, groups, rhos, parents = [], [], [], []
        for k, (g, R, Z, W, H, a1, a2, rho) in enumerate(raw_elements(exclude)):
            for p in _split(R, Z, W, H, a1, a2, piece_len):
                pieces.append(p); groups.append(g); rhos.append(rho); parents.append(k)
        self.piece = pieces
        self.groups = np.array(groups)
        self.parent = np.array(parents)
        self.rho = np.array(rhos)
        self.Rc = np.array([p[0] for p in pieces]); self.Zc = np.array([p[1] for p in pieces])
        self.report = np.array([report_group(g, z) for g, z in zip(groups, self.Zc)])
        self.area = np.array([p[2] * p[3] for p in pieces])
        geo = machine_geometry("mast_public_test")
        self.gR, self.gZ = geo["grid_R"], geo["grid_Z"]
        key = hashlib.sha1(json.dumps([pieces, rhos, self.coils, self.sol_fraction, plasma,
                                       self.gR.tolist(), self.gZ.tolist()]).encode()
                           ).hexdigest()[:12]
        f = OUT / "model" / f"passive_model_{key}.npz"
        self.cache_file = f
        if cache and f.exists():
            z = np.load(f)
            self.Rres, self.L, self.Mpc, self.MpP, self.Gg = (z["Rres"], z["L"], z["Mpc"],
                                                               z["MpP"], z["Gg"])
            return
        self._build(geo)
        if cache:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez(f, Rres=self.Rres, L=self.L, Mpc=self.Mpc, MpP=self.MpP, Gg=self.Gg)

    # ------------------------------------------------------------------ build matrices
    def _build(self, geo):
        Np = len(self.piece)
        fr, fz, fw, seg = [], [], [], []
        G = np.zeros(Np)
        for i, (R, Z, W, H, a1, a2) in enumerate(self.piece):
            r, z, w = element_filaments(R, Z, W, H, a1, a2)
            fr.append(r); fz.append(z); fw.append(w); seg.append(np.full(r.size, i))
            # conductance of the piece: sum of filament tubes in parallel
            G[i] = np.sum(w * self.area[i] / (2 * np.pi * r)) / self.rho[i]
        fr, fz, fw, seg = map(np.concatenate, (fr, fz, fw, seg))
        self.Rres = 1.0 / G
        # mutual inductance between pieces: average psi over the receiving piece's filaments
        P = _psi_points(fr, fz, fr, fz, fw, seg, Np)            # (Nf, Np)
        A = np.zeros((Np, fr.size)); A[seg, np.arange(fr.size)] = fw
        M = 2 * np.pi * (A @ P)
        M = 0.5 * (M + M.T)
        # self-inductance: L = mu0 R (ln(8R/GMD) - 2), GMD of a rectangle = 0.2235 (a+b)
        W = np.array([p[2] for p in self.piece]); H = np.array([p[3] for p in self.piece])
        gmd = 0.2235 * (W + H)
        M[np.arange(Np), np.arange(Np)] = MU0 * self.Rc * (np.log(8 * self.Rc / gmd) - 2.0)
        self.L = M
        # coils -> pieces (per unit input-column current, as fec.machines.mast_setup)
        cr, cz, cw, cs = [], [], [], []
        counts = {c: 0 for c in self.coils}
        for c in geo["cols"]:
            if c in counts:
                counts[c] += 1
        for (R, Z, W, H, a1, a2), c in zip(geo["elements"], geo["cols"]):
            if c not in counts:
                continue
            j = self.coils.index(c)
            r, z, w = element_filaments(R, Z, W, H, a1, a2)
            per_turn = self.sol_fraction if c == "magnetics_sol_current" else 1.0 / counts[c]
            cr.append(r); cz.append(z); cw.append(w * per_turn); cs.append(np.full(r.size, j))
        cr, cz, cw, cs = map(np.concatenate, (cr, cz, cw, cs))
        Pc = _psi_points(fr, fz, cr, cz, cw, cs, len(self.coils))  # (Nf, C)
        self.Mpc = 2 * np.pi * (A @ Pc)
        pr, pz, pw = nominal_plasma(self.plasma)
        Pp = _psi_points(fr, fz, pr, pz, pw, np.zeros(pr.size, int), 1)
        self.MpP = 2 * np.pi * (A @ Pp)[:, 0]
        RR, ZZ = np.meshgrid(geo["grid_R"], geo["grid_Z"])
        self.Gg = _psi_points(RR.ravel(), ZZ.ravel(), fr, fz, fw, seg, Np)

    # ------------------------------------------------------------------ dynamics
    def rvec(self, rscale=1.0):
        """Piece resistances with a global float or {report-group-prefix: factor} scaling."""
        if isinstance(rscale, dict):
            f = np.ones(len(self.piece))
            for k, v in rscale.items():
                f[np.char.startswith(self.report.astype(str), k)] *= v
            return self.Rres * f
        return self.Rres * float(rscale)

    def modes(self, rscale=1.0):
        """Normal modes: tau (Np,) [s], V (Np,Np) with I = V x, W (Np,Np) with f = W u."""
        R = self.rvec(rscale)
        s = 1.0 / np.sqrt(R)
        S = s[:, None] * self.L * s[None, :]
        tau, Q = np.linalg.eigh(0.5 * (S + S.T))
        if tau.min() <= 0:
            raise ValueError(f"inductance matrix not positive definite (min tau {tau.min()})")
        V = s[:, None] * Q
        W = Q.T * s[None, :]
        return tau, V, W

    def drive(self, Ic, Ip):
        """External flux Phi [Wb] at each piece: Ic (...,C) [A], Ip (...) [A] -> (...,Np)."""
        return Ic @ self.Mpc.T + np.asarray(Ip)[..., None] * self.MpP

    def integrate(self, t_s, Ic, Ip, rscale=1.0, modes=None):
        """C1: currents I (T,Np) [A] from I=0 at t_s[0]; exact for piecewise-linear drive."""
        tau, V, W = self.modes(rscale) if modes is None else modes
        f = self.drive(Ic, Ip) @ W.T                     # (T, Np) modal drive
        x = np.zeros((len(t_s), len(tau)))
        xi = np.zeros(len(tau))
        for n in range(len(t_s) - 1):
            h = t_s[n + 1] - t_s[n]
            a = np.exp(-h / tau)
            xi = a * xi - (1 - a) * (f[n + 1] - f[n]) / h
            x[n + 1] = xi
        return x @ V.T

    def quasi_steady(self, dIc_dt, dIp_dt, rscale=1.0):
        """C5: I = -(dPhi/dt)/R per piece (no mutual/self inductance)."""
        return -self.drive(dIc_dt, dIp_dt) / self.rvec(rscale)

    # ------------------------------------------------------------------ outputs
    def psi_grid(self, I):
        """psi_passive (..., 65, 65) [Wb/rad] from piece currents I (..., Np)."""
        I = np.asarray(I)
        return (I @ self.Gg.T).reshape(I.shape[:-1] + (len(self.gZ), len(self.gR)))

    def augment(self, M):
        """Coil matrix with one extra 'coil' column per passive piece: psi_ext + psi_passive
        = [Ic, I_passive] @ augment(M).T  (use with fec.gs.Solver(spec, augment(M)))."""
        return np.concatenate([M, self.Gg], 1)

    def group_totals(self, I):
        """{report group: total current (...)} plus 'total'."""
        I = np.asarray(I)
        out = {g: I[..., self.report == g].sum(-1) for g in np.unique(self.report)}
        out["total"] = I.sum(-1)
        return out

    def table(self, rscale=1.0):
        """Per raw-element summary rows for documentation."""
        R = self.rvec(rscale)
        rows = []
        for g in dict.fromkeys(self.report):
            m = self.report == g
            rows.append(dict(group=str(g), pieces=int(m.sum()),
                             R_range=(float(self.Rc[m].min()), float(self.Rc[m].max())),
                             Z_range=(float(self.Zc[m].min()), float(self.Zc[m].max())),
                             area=float(self.area[m].sum()),
                             resistance=float(1.0 / np.sum(1.0 / R[m]))))
        return rows


# ----------------------------------------------------------------------------- signals
SIG_CH = [f"magnetics_p{i}{s}_current" for i in range(2, 7) for s in "lu"] + [
    "magnetics_sol_current", "magnetics_plasma_current"]
T0_MS, DT_MS = -300.0, 0.5


def _signals_one(args):
    """Parquet (inputs only) -> 0.5 ms box-mean coil/Ip currents from -300 ms to last EFIT time."""
    import pyarrow.parquet as pq
    from fec.extract import box_means, col1d
    path, outdir = args
    path = Path(path)
    out = Path(outdir) / (path.stem + ".npz")
    if out.exists():
        return path.stem, "skip"
    t = pq.read_table(path, columns=["magnetics_time", "efit_times"] + SIG_CH)
    mt = col1d(t, "magnetics_time")
    et = col1d(t, "efit_times")
    tg = np.arange(T0_MS, et.max() + DT_MS + 1e-9, DT_MS)
    sig = np.stack([box_means(mt, col1d(t, c), tg, half=DT_MS) for c in SIG_CH], 1)
    tmp = out.with_suffix(".part.npz")
    np.savez(tmp, t_ms=tg, sig=sig.astype(np.float32), channels=np.array(SIG_CH),
             efit_times=et)
    tmp.replace(out)
    return path.stem, "ok"


def extract_signals(workers=6, config="mast_public_test"):
    from fec.paths import shot_files
    files = shot_files(config)
    outdir = OUT / "sig" / config
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for i, (stem, st) in enumerate(pool.imap_unordered(
                _signals_one, [(str(f), str(outdir)) for f in files], chunksize=4)):
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    print(f"signals done {len(files)} in {time.time() - t0:.0f}s", flush=True)


def frame_index(config="mast_public_test"):
    """ds frame order -> per shot stem the list of (global frame index, t_ms)."""
    from fec.features import load_ds
    ds = load_ds(config)
    order = ds["info"]["order"]
    per = {}
    for g, (s, t) in enumerate(zip(ds["shot"], ds["t_ms"])):
        per.setdefault(order[s], []).append((g, float(t)))
    return per, len(ds["t_ms"]), ds


def c1_shot(model, z, rscale=1.0, modes=None):
    """Integrate one shot's signal file z -> (t_ms, I (T,Np))."""
    from fec.machines import MAST_COILS
    ch = list(z["channels"])
    sig = np.asarray(z["sig"], dtype=np.float64)
    sig = np.where(np.isfinite(sig), sig, 0.0)
    Ic = np.stack([sig[:, ch.index(c)] for c in MAST_COILS], 1) * 1e3
    Ip = sig[:, ch.index("magnetics_plasma_current")] * 1e3
    t = np.asarray(z["t_ms"], dtype=np.float64)
    return t, model.integrate(t * 1e-3, Ic, Ip, rscale=rscale, modes=modes)


def run_frames(model_kind="c1", rscale=1.0, save_psi=False, tag=None, plasma="ellipse",
               config="mast_public_test"):
    """Per-frame passive piece currents aligned with FEC_WORK/ds/<config> frame order."""
    from fec.machines import MAST_CH, MAST_COILS
    model = PassiveModel(plasma=plasma)
    tag = tag or f"{model_kind}_r{rscale:g}" + ("" if plasma == "ellipse" else f"_{plasma}")
    od = OUT / "frames" / config / tag
    od.mkdir(parents=True, exist_ok=True)
    per, N, ds = frame_index(config)
    I_out = np.lib.format.open_memmap(od / "I.npy", "w+", np.float32, (N, len(model.piece)))
    t0 = time.time()
    if model_kind == "c1":
        modes = model.modes(rscale)
        sd = OUT / "sig" / config
        for k, (stem, fr) in enumerate(per.items()):
            z = np.load(sd / f"{stem}.npz")
            t, I = c1_shot(model, z, rscale, modes)
            g = np.array([a for a, _ in fr]); tf = np.array([b for _, b in fr])
            for j in range(I.shape[1]):
                I_out[g, j] = np.interp(tf, t, I[:, j])
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(per)} shots {time.time() - t0:.0f}s", flush=True)
        tau = modes[0]
    elif model_kind == "c5":
        cur = np.asarray(ds["cur"], dtype=np.float64)       # (N, C, K) kA
        offs = ds["info"]["offsets_ms"]
        kp, km = offs.index(5.0), offs.index(-5.0)
        ci = [MAST_CH.index(c) for c in MAST_COILS]
        ip = MAST_CH.index("magnetics_plasma_current")
        dt = (offs[kp] - offs[km]) * 1e-3
        dIc = np.nan_to_num((cur[:, ci, kp] - cur[:, ci, km]) * 1e3 / dt)
        dIp = np.nan_to_num((cur[:, ip, kp] - cur[:, ip, km]) * 1e3 / dt)
        for s0 in range(0, N, 8192):
            sl = slice(s0, min(N, s0 + 8192))
            I_out[sl] = model.quasi_steady(dIc[sl], dIp[sl], rscale)
        tau = model.modes(rscale)[0]
    else:
        raise ValueError(model_kind)
    I_out.flush()
    tot = model.group_totals(np.asarray(I_out))
    np.savez(od / "totals.npz", **tot)
    meta = dict(model=model_kind, rscale=rscale, plasma=plasma, n_pieces=len(model.piece),
                model_cache=str(model.cache_file), tau_max_ms=float(tau.max() * 1e3),
                tau_top5_ms=(np.sort(tau)[::-1][:5] * 1e3).tolist(), seconds=time.time() - t0)
    (od / "meta.json").write_text(json.dumps(meta, indent=1))
    if save_psi:
        P = np.lib.format.open_memmap(od / "psi_passive.npy", "w+", np.float32,
                                      (N, len(model.gZ), len(model.gR)))
        for s0 in range(0, N, 4096):
            sl = slice(s0, min(N, s0 + 4096))
            P[sl] = model.psi_grid(np.asarray(I_out[sl], dtype=np.float64))
        P.flush()
    print(json.dumps(meta), flush=True)
    return od


def summary():
    m = PassiveModel()
    print(f"pieces {len(m.piece)}  cache {m.cache_file}")
    for r in m.table():
        print(f"{r['group']:16s} n={r['pieces']:3d} R {r['R_range'][0]:.3f}-{r['R_range'][1]:.3f}"
              f" Z {r['Z_range'][0]:+.3f}..{r['Z_range'][1]:+.3f} A={r['area'] * 1e4:8.2f} cm2"
              f" Rpar={r['resistance']:.3e} ohm")
    tau = m.modes()[0]
    print("slowest L/R modes [ms]:", np.round(np.sort(tau)[::-1][:8] * 1e3, 2).tolist())
    ev = np.linalg.eigvalsh(m.L)
    print(f"L eigen min/max {ev.min():.3e} {ev.max():.3e} H")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["summary", "signals", "frames"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default="c1", choices=["c1", "c5"])
    ap.add_argument("--rscale", type=float, default=1.0)
    ap.add_argument("--plasma", default="ellipse", choices=["ellipse", "filament"])
    ap.add_argument("--psi", action="store_true")
    ap.add_argument("--config", default="mast_public_test")
    a = ap.parse_args()
    if a.cmd == "summary":
        summary()
    elif a.cmd == "signals":
        extract_signals(a.workers, a.config)
    else:
        run_frames(a.model, a.rscale, a.psi, plasma=a.plasma, config=a.config)
