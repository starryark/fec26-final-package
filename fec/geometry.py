"""Machine coil geometry read from the shipped parquet columns (inputs, allowed)."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq

from fec.paths import WORK, shot_files

GEOM_COLS = ["coil_name", "coil_input_column", "coil_R", "coil_Z", "coil_width", "coil_height",
             "coil_angle1", "coil_angle2", "efit_grid_R", "efit_grid_Z"]


@lru_cache(maxsize=4)
def machine_geometry(config):
    cache = WORK / "geometry" / f"{config}.json"
    if cache.exists():
        g = json.loads(cache.read_text())
    else:
        f = shot_files(config)[0]
        t = pq.read_table(f, columns=GEOM_COLS)
        g = {c: t.column(c)[0].as_py() for c in GEOM_COLS}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(g))
    elements = list(zip(g["coil_R"], g["coil_Z"], g["coil_width"], g["coil_height"],
                        g["coil_angle1"], g["coil_angle2"]))
    return {"names": g["coil_name"], "cols": g["coil_input_column"], "elements": elements,
            "grid_R": np.array(g["efit_grid_R"]), "grid_Z": np.array(g["efit_grid_Z"])}


def check_geometry_constant(config, n=20):
    """Verify that coil geometry and grid are identical across a sample of shots."""
    ref = None
    fs = shot_files(config)
    idx = np.linspace(0, len(fs) - 1, n).astype(int)
    for i in idx:
        t = pq.read_table(fs[i], columns=GEOM_COLS)
        g = {c: t.column(c)[0].as_py() for c in GEOM_COLS}
        if ref is None:
            ref = g
        elif g != ref:
            return False, fs[i].name
    return True, None
