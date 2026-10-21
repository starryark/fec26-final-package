"""Machine specs + external-field matrices for the GS solver."""
from __future__ import annotations

import json

import numpy as np

from fec.geometry import machine_geometry
from fec.green import coil_psi_matrix
from fec.gs import MachineSpec
from fec.paths import WORK

MAST_CH = [f"magnetics_p{i}{s}_current" for i in range(2, 7) for s in "lu"] + [
    "magnetics_sol_current", "magnetics_tf_current", "magnetics_efps_current",
    "magnetics_plasma_current"]
MAST_COILS = [c for c in MAST_CH if c.startswith("magnetics_p") or c == "magnetics_sol_current"]
MAST_TF_TURNS = 24  # MAST toroidal field coil: 24 turns in series (machine description)
# MAST drives its 4-layer solenoid as two parallel circuits: each of the 656 turns carries
# half the feed current (Battaglia et al. UKAEA-CCFE-PR(19)34; 656*I/2 reproduces the
# published 0.8-0.9 Vs swing). See docs/research/MAST_CONVENTIONS_RESEARCH_20261007.md.
MAST_SOL_TURN_FRACTION = 0.5


def mast_setup(sol_fraction=MAST_SOL_TURN_FRACTION):
    g = machine_geometry("mast_public_test")
    f = WORK / "geometry" / ("mast_M.npy" if sol_fraction == 1.0 else f"mast_M_sol{sol_fraction:g}.npy")
    if f.exists():
        M = np.load(f)
    else:
        M, counts = coil_psi_matrix(g["grid_R"], g["grid_Z"], g["elements"], g["cols"], MAST_COILS)
        # solenoid: sol_current is per turn; each of its elements is one turn
        isol = MAST_COILS.index("magnetics_sol_current")
        M[:, isol] *= counts[isol] * sol_fraction
        np.save(f, M)
    keep = []
    for (R, Z, W, H, a1, a2), c in zip(g["elements"], g["cols"]):
        if c.startswith("magnetics_p2") or c.startswith("magnetics_p3") or c.startswith("magnetics_p6"):
            keep.append((R, Z, 0.07))
    spec = MachineSpec("mast", g["grid_R"], g["grid_Z"], box=(0.20, 1.56, -1.62, 1.62),
                       axis_box=(0.45, 1.3, -0.6, 0.6), R0=0.85, keepout=keep)
    return spec, M


def mast_currents(cur_row):
    """cur_row: (C,) kA values in MAST_CH order -> (coil ampere array, Ip [A], tf [kA])."""
    I = np.array([cur_row[MAST_CH.index(c)] for c in MAST_COILS]) * 1e3
    return I, cur_row[MAST_CH.index("magnetics_plasma_current")] * 1e3, \
        cur_row[MAST_CH.index("magnetics_tf_current")]
