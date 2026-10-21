"""NRS-L cascade stage of the MAST half (configs final_*_nrsl; family doc amendment 68; agent nrsl_pkg 2026-10-20).

Shipped with the package.  Research-tree-only checks (the verbatim copies against their source modules, the package
modules against mast_solver2's frozen tools, the docstring scrub of fec/current_diffusion.py) skip where those files are
absent.  Covers G1 (code identity, settings), G9 (batch partition, fallback rule, increment-smoothing linearity,
untouched scalars), the recipes and the stage wiring.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# mast_solver2 INTEGRATION_SPEC §3 (the frozen stage settings) and §7 (the label-free fold assertions)
SPEC_STAGES = {"L": {"rmode": "pinb", "kp": 5.0, "k_li": 2.0, "dv_clamp": 0.01, "kr": 0.02, "li": True},
               "P": {"rmode": "pin", "kp": 3.0, "k_li": 2.0, "dv_clamp": 0.01, "kr": 0.02, "li": False}}
SPEC_ASSERTS = {"lcfs_found_min": 0.99, "inside_wall_min": 0.85, "ip_err_median_max": 0.03, "absZ_gt_0p3_max": 0.01,
                "nonfinite_max": 0, "axis_median_cm_max": 1.0, "axis_frac_gt3cm_max": 0.05, "li_coverage_min": 0.99,
                "li_inrange_min": 0.95, "li_clamp_max": 0.30}
# SHA-256 of every config file of the v11 package (pkg_v11 476bbf08, configs/<name>.json): unchanged in v12
V11_CONFIG_SHA256 = {
    "final_A22li": "afaf9b9536962c0bdcf21ff6370c9c5bed06648aaed1a603caa927a43c35ad55",
    "final_A22li_dec": "975a6d0a224fb967acf07f84d2f9b7312f4eb0915a5843501093d7e6f2860b58",
    "final_A22li_dec_mmk": "7609028312e40ddd9ab043dc9265e09b6d1db3a390d77e5f15e3ca457e2db61d",
    "final_A22li_r05": "848f3ffea75cb80cd8ffe062c2683e8ee04ba828d1bf6285f941fc2df575847e",
    "final_A22li_r2": "59435535c8eef55502ba7fb6517baa60a2690d786f14848d470f6f25dcf352be",
    "final_A24": "82d530099fa51faeaf6e7d10438226c33d956c2bc233f44f5b06ba0a6bd8c42c",
    "final_A24r2": "8262408d0b739cce92cf583ff49976a27accc0f28f87bfc40e7b03a05cff5498",
    "final_B": "71f1c84607fed29e3eb751cf5b2a1f8380b9e905e6306ba16d4024d9f492cec3",
    "final_C": "c1ac8a962391cc862adb97a3e3309f2d683c6695735245c7d1c8ca3aa0041bcf",
    "final_MTfull": "5c626050a8349f1dc6579aa318cb2cf2782ed444dd31f24d16d6f84ab4e0fb48",
    "final_MTfull_mmk": "af28b59de7355b5991caaf79af13f1cb2949e98481ffbbe19dfa3d73e42dffdf",
    "final_MTli": "0ab65610b02f52d4fe23eb46f700764306985b6f84f37e0a10395e7a9403e134"}
NRSL = ("final_MTfull_mmk_nrsl", "final_A22li_dec_mmk_nrsl")
PARENT = {"final_MTfull_mmk_nrsl": "final_MTfull_mmk", "final_A22li_dec_mmk_nrsl": "final_A22li_dec_mmk"}
# mast_solver2's frozen tools -> package modules, and the reviewed import/path lines that differ (G1)
FROZEN_TOOLS = {"fec/nrsl_op.py": ("ms_op.py", 1), "fec/nrsl_cascade.py": ("ms_cascade.py", 0),
                "fec/nrsl_run.py": ("ms2_mast_run.py", 2), "fec/nrsl_li.py": ("ms2_mast_li.py", 7),
                "fec/nrsl_post.py": ("ms2_mast_post.py", 3), "fec/nrsl_expand.py": ("ms2_expand.py", 0)}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _RF():
    import fec.run_final as RF
    return RF


class TestCodeIdentity(unittest.TestCase):
    """G1: pinned stage files, settings = spec §3, assertions = spec §7, the li estimator's frozen code."""

    def test_recipe_pins_every_stage_file(self):
        RF = _RF()
        q = RF.NRSL_CASCADE
        self.assertEqual(set(q["code_sha256"]), {"fec/nrsl_op.py", "fec/nrsl_cascade.py", "fec/nrsl_run.py",
                                                 "fec/nrsl_li.py", "fec/nrsl_post.py", "fec/nrsl_expand.py",
                                                 "fec/nrsl_copies.py", "fec/mast_nrsl.py"})
        for rel, h in q["code_sha256"].items():
            self.assertEqual(sha_file(REPO / rel), h, rel)

    def test_stage_settings_equal_spec(self):
        RF = _RF()
        from fec import nrsl_run
        self.assertEqual(json.loads(json.dumps(nrsl_run.STAGES)), SPEC_STAGES)
        q = RF.NRSL_CASCADE
        self.assertEqual(q["stages"], SPEC_STAGES)
        self.assertEqual((q["device"], q["threads"], q["batch"], q["n_iter"], q["sol_frac"], q["ip_floor_A"],
                          q["increment_tsm_sigma_ms"], q["cascade"]),
                         ("cpu", 8, 256, 150, 0.55, 1e5, 30.0, ["L", "P", "base"]))
        self.assertEqual(q["env"], {"MKL_CBWR": "AVX2", "OPENBLAS_CORETYPE": "Haswell"})
        self.assertEqual(q["asserts"], SPEC_ASSERTS)
        from fec import nrsl_op
        self.assertEqual((nrsl_op.DV_CLAMP, nrsl_op.SP_LO, nrsl_op.SP_HI, nrsl_op.GAM_CLAMP, nrsl_op.LI_TOL,
                          nrsl_op.CONV_TOL, nrsl_op.MISS_TOL), (0.010, -0.9, 4.0, 4.0, 0.03, 1e-2, 0.01))

    def test_post_bars_equal_recipe_asserts(self):
        from fec import nrsl_post
        q = _RF().NRSL_CASCADE["asserts"]
        self.assertEqual(nrsl_post.BARS, dict(lcfs=q["lcfs_found_min"], inside=q["inside_wall_min"],
                                              iperr=q["ip_err_median_max"], zfrac=q["absZ_gt_0p3_max"]))

    def test_li_estimator_frozen_code(self):
        from fec import current_diffusion as CD
        from fec.nrsl_copies import CD_SHA256
        q = _RF().NRSL_CASCADE["li"]
        self.assertIn(sha_file(CD.__file__), CD_SHA256.values())
        self.assertEqual(sorted(CD_SHA256.values()), sorted(q["current_diffusion_sha256"]))

    def test_package_copy_of_current_diffusion_is_the_scrub(self):
        """The one-line difference between the frozen estimator and the package's copy: the build's docstring scrub
        of line 77 (outputs path).  Research tree only (fec/final_pkg.py is the builder)."""
        from fec import current_diffusion as CD
        from fec.nrsl_copies import CD_SHA256
        if not (REPO / "fec" / "final_pkg.py").exists() or sha_file(CD.__file__) != CD_SHA256["source"]:
            self.skipTest("research tree with the frozen estimator only")
        from fec.final_pkg import code_ast, scrub_text
        src = Path(CD.__file__).read_bytes().decode("utf-8")
        new = scrub_text(src)
        self.assertEqual(hashlib.sha256(new.encode("utf-8")).hexdigest(), CD_SHA256["package"])
        self.assertEqual(code_ast(new), code_ast(src))
        diff = [(a, b) for a, b in zip(src.splitlines(), new.splitlines()) if a != b]
        self.assertEqual(len(diff), 1)
        self.assertEqual(diff[0][1], "Outputs: <FEC_WORK>/cdiff/ (d3d_<set>.npz, mast.npz, frozen.json, *_report.json).")

    def test_package_modules_vs_frozen_tools(self):
        """Research tree only: each package module = mast_solver2's frozen tool except the reviewed import/path lines
        (sys.path, module names, the li target's package sha assert and parquet location)."""
        from fec.paths import WORK
        tools = WORK / "scratch" / "mast_solver_20261020" / "tools"
        if not tools.is_dir():
            self.skipTest("frozen tools not on this machine")
        allowed = ("sys.path.insert(", "import ", "from fec", "assert _sha(CD.__file__)", "jobs, slices",
                   "jobs.append((stem,", "r = subprocess.run([sys.executable, \"-B\", str(HERE / ")
        for rel, (src, n_edit) in FROZEN_TOOLS.items():
            a = (tools / src).read_bytes()
            b = (REPO / rel).read_bytes()
            self.assertEqual(b"\r\n" in a, b"\r\n" in b, rel)
            la, lb = a.decode().splitlines(), b.decode().splitlines()
            self.assertEqual(len(la), len(lb), rel)
            diff = [(x, y) for x, y in zip(la, lb) if x != y]
            self.assertEqual(len(diff), n_edit, (rel, diff))
            for x, y in diff:
                self.assertTrue(x.strip().startswith(allowed) and y.strip().startswith(allowed), (rel, x, y))


class TestVerbatimCopies(unittest.TestCase):
    """fill_in_shot, meta_from_maps and the veto helpers are verbatim copies of their research-tree sources."""

    def _check(self, mod, names):
        import importlib
        try:
            src = importlib.import_module(mod)
        except ImportError:
            self.skipTest(f"{mod} not in this tree (package)")
        from fec import nrsl_copies as C
        for n in names:
            a, b = getattr(C, n), getattr(src, n)
            if callable(a):
                self.assertEqual(inspect.getsource(a), inspect.getsource(b), n)
            else:
                self.assertEqual(a, b, n)

    def test_fill_in_shot(self):
        self._check("fec.licond_d3d", ["FROZEN_SHA_PREFIX", "fill_in_shot"])

    def test_meta_from_maps(self):
        self._check("fec.amort_real", ["meta_from_maps"])

    def test_vetoes(self):
        self._check("fec.xfer_eval", ["MU0", "_GEO", "_mast_geo", "delta_star", "dist_to_polygon", "_g3_frame"])

    def test_fill_rule_behaviour(self):
        from fec.nrsl_copies import fill_in_shot
        x = np.array([np.nan, 1.0, np.nan, 3.0, np.nan, np.nan])
        shot = np.array([0, 0, 0, 0, 1, 1])
        t = np.array([0.0, 10.0, 24.0, 30.0, 0.0, 5.0])
        y, f = fill_in_shot(x, shot, t, 9.0)
        self.assertEqual(y.tolist(), [1.0, 1.0, 3.0, 3.0, 9.0, 9.0])
        self.assertEqual(f.tolist(), [True, False, True, False, True, True])


class TestRecipes(unittest.TestCase):
    def test_nrsl_recipe_is_parent_plus_stage(self):
        RF = _RF()
        for n in NRSL:
            a, b = json.loads(json.dumps(RF.RECIPES[n])), json.loads(json.dumps(RF.RECIPES[PARENT[n]]))
            q = a["mast"]["composite"].pop("nrsl")
            self.assertEqual(q, RF.NRSL_CASCADE)
            self.assertEqual(a["d3d"], b["d3d"], n)
            self.assertEqual(a["mast"], b["mast"], n)
            self.assertEqual({k: v for k, v in a.items() if k != "name"}, {k: v for k, v in b.items() if k != "name"})
            self.assertEqual(a["name"], b["name"].replace("r2tsm30_", "r2tsm30nrsl_"))
        for n, r in RF.RECIPES.items():
            if n not in NRSL:
                self.assertNotIn("nrsl", (r.get("mast") or {}).get("composite") or {}, n)

    def test_stage_order(self):
        RF = _RF()
        for n in NRSL:
            st = RF.all_stages(RF.RECIPES[n])
            i = st.index("mast_physics")
            self.assertEqual(st[i + 1], "mast_nrsl")
            self.assertLess(st.index("mast_nrsl"), st.index("mast_composite"))
            self.assertEqual([s for s in st if s != "mast_nrsl"], RF.all_stages(RF.RECIPES[PARENT[n]]))
        for n, r in RF.RECIPES.items():
            if n not in NRSL:
                try:
                    self.assertNotIn("mast_nrsl", RF.all_stages(r), n)
                except ValueError:
                    pass

    def test_load_recipe_validates_block(self):
        RF = _RF()
        with tempfile.TemporaryDirectory() as d:
            for bad in ({"device": "cuda"}, {"increment_tsm_sigma_ms": 50.0}, {"sol_frac": 0.5}, {"stages": {"L": {}}},
                        {"processes": 0}):
                r = json.loads(json.dumps(RF.RECIPES["final_MTfull_mmk_nrsl"]))
                r["mast"]["composite"]["nrsl"].update(bad)
                f = Path(d) / "r.json"
                f.write_text(json.dumps(r))
                with self.assertRaises(ValueError, msg=str(bad)):
                    RF.load_recipe(str(f), d)
            RF.load_recipe("final_MTfull_mmk_nrsl", d)

    def test_v11_configs_unchanged(self):
        """Every v11 config is written byte for byte as in the v11 package (only the two NRS-L configs are new)."""
        if not (REPO / "fec" / "final_pkg.py").exists():          # the package: its configs/ files
            for n, h in V11_CONFIG_SHA256.items():
                self.assertEqual(sha_file(REPO / "configs" / f"{n}.json"), h, n)
            self.assertEqual(sorted(p.stem for p in (REPO / "configs").glob("final_*.json")),
                             sorted(list(V11_CONFIG_SHA256) + list(NRSL)))
            return
        from fec import final_pkg as FP
        for n, h in V11_CONFIG_SHA256.items():
            s = (json.dumps(FP.recipe_for_package(n), indent=1) + chr(10)).encode("utf-8")
            self.assertEqual(hashlib.sha256(s).hexdigest(), h, n)
        self.assertEqual(sorted(FP.package_cands()), sorted(list(V11_CONFIG_SHA256) + list(NRSL)))

    def test_package_configs(self):
        RF = _RF()
        if not (REPO / "fec" / "final_pkg.py").exists():          # the package: the shipped config files
            for n in NRSL:
                cfg = json.loads((REPO / "configs" / f"{n}.json").read_text(encoding="utf-8"))
                self.assertEqual(cfg["mast"]["composite"]["nrsl"], RF.NRSL_CASCADE)
            return
        from fec import final_pkg as FP
        for n in NRSL:
            self.assertIn(n, FP.package_cands())
            self.assertEqual(FP.CAND_GROUPS[n], FP.CAND_GROUPS[PARENT[n]])
            cfg = FP.recipe_for_package(n)
            self.assertEqual(cfg["mast"]["composite"]["nrsl"], RF.NRSL_CASCADE)
        mods = FP.closure()
        for m in ("mast_nrsl", "nrsl_op", "nrsl_cascade", "nrsl_run", "nrsl_li", "nrsl_post", "nrsl_expand",
                  "nrsl_copies", "current_diffusion", "betap_li", "betan_transfer", "mast_tsm"):
            self.assertIn(m, mods)
        for m in ("amort_real", "licond_d3d", "xfer_eval"):
            self.assertNotIn(m, mods)
        self.assertEqual(FP.FROZEN["frozen/cdiff/frozen.json"][1], "fec_work/cdiff/frozen.json")
        self.assertEqual(FP.FROZEN["frozen/licond/sigma.json"][1], "fec_work/licond/sigma.json")


def _args(**kw):
    base = dict(split="private_test", subset_shots=0, root=None, dry_run=True, stage="all", d3d_work=None,
                owner="test", gpu_priority="normal", no_runlog=True, dec_dir=None, d3d_dir=None, q95h_src=None,
                force=False, workers=5, allow_unreviewed=False, no_model_hash=True, mast_dir=None,
                mast_scalars_dir=None, amort_net_dir=None, psi_dtype=None, submission_name="submission",
                mast_jobs=1)
    base.update(kw)
    return SimpleNamespace(**base)


class TestUntouchedScalars(unittest.TestCase):
    """Only MAST psirz changes: the composite takes its maps from the stage node and everything else (scalars node,
    betaN, submission command) exactly as the parent config does."""

    def _cmds(self, name, stages, d):
        RF = _RF()
        c = RF.Ctx(_args(root=d), json.loads(json.dumps(RF.RECIPES[name])))
        for s in stages:
            RF.STAGES[s][0](c)
        return c, [x.split(" -- ")[-1] for x in c.cmds if not x.startswith("link ")]

    def test_composite_and_submission_commands(self):
        RF = _RF()
        with tempfile.TemporaryDirectory() as d:
            for n in NRSL:
                cp, par = self._cmds(PARENT[n], ["mast_composite"], d)
                cn, new = self._cmds(n, ["mast_nrsl", "mast_composite"], d)
                Dp, Dn = RF.composite_dirs(cp), RF.composite_dirs(cn)
                self.assertEqual(Dn["maps"], Dn["nrsl"])
                self.assertEqual(Dn["nrsl"].name, Dp["maps"].name + "_nrsl")
                self.assertEqual({k: v for k, v in Dn.items() if k not in ("maps", "nrsl")},
                                 {k: v for k, v in Dp.items() if k != "maps"})
                self.assertEqual(RF.composite_scalars_dir(cn), RF.composite_scalars_dir(cp))
                self.assertEqual(len(new), 2)
                self.assertIn("mast_nrsl.py run --config mast_private_test", new[0])
                for k in (f"--maps {Dp['maps']}", f"--out {Dn['nrsl']}", "--workers 5", "--procs 1",
                          f"{Dp['passive']}{os.sep}c1_r2{os.sep}I.npy"):
                    self.assertIn(k, new[0].replace('"', ""))
                self.assertEqual(new[1].replace(str(Dn["maps"]), "MAPS"), par[0].replace(str(Dp["maps"]), "MAPS"))
                self.assertEqual(self._cmds(n, ["submission"], d)[1], self._cmds(PARENT[n], ["submission"], d)[1])

    def test_stage_env(self):
        RF = _RF()
        from fec import mast_nrsl
        e = RF.nrsl_env(RF.NRSL_CASCADE)
        self.assertEqual(e, dict(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8",
                                 MKL_CBWR="AVX2", OPENBLAS_CORETYPE="Haswell"))
        self.assertEqual({k: v for k, v in mast_nrsl.stage_env(RF.NRSL_CASCADE).items() if k != "PYTHONUNBUFFERED"}, e)


class TestBatchPartition(unittest.TestCase):
    """The process split keeps every 256-frame batch identical to the one-process run's batch (G5's precondition)."""

    def test_chunks_are_whole_batches(self):
        from fec.mast_nrsl import frame_chunks
        for n in (1, 255, 256, 257, 1000, 68373):
            for procs in (1, 2, 3, 12):
                fr = np.arange(n) * 3 + 7
                ch = frame_chunks(fr, procs, 256)
                self.assertLessEqual(len(ch), procs)
                self.assertTrue(np.array_equal(np.concatenate(ch), fr))
                for c in ch[:-1]:
                    self.assertEqual(len(c) % 256, 0)
                full = [tuple(fr[s:s + 256]) for s in range(0, n, 256)]
                parts = [tuple(c[s:s + 256]) for c in ch for s in range(0, len(c), 256)]
                self.assertEqual(parts, full)
        self.assertEqual(frame_chunks(np.arange(0), 3, 256), [])

    def test_run_stage_merges_in_order(self):
        """run_stage with K processes: a fake nrsl_run (output = a function of each batch's own frames) gives the same
        merged file as one process."""
        from fec import mast_nrsl as M
        fake = (
            "import sys, numpy as np\n"
            "a = sys.argv; d = a[a.index('--fold-dir') + 1]; out = a[a.index('--out') + 1]; bs = int(a[a.index('--bs') + 1])\n"
            "N = np.load(d + '/base.npy', mmap_mode='r').shape[0]\n"
            "fr = np.load(a[a.index('--frames') + 1]) if '--frames' in a else np.arange(N)\n"
            "psi = np.zeros((len(fr), 2), np.float32); use = np.zeros(len(fr), bool)\n"
            "for s in range(0, len(fr), bs):\n"
            "    b = fr[s:s + bs]; psi[s:s + len(b), 0] = b.sum() + np.arange(len(b)); psi[s:s + len(b), 1] = len(b)\n"
            "    use[s:s + len(b)] = (b % 3) == 0\n"
            "np.savez(out, frames=fr, psi=psi, use=use)\n")
        q = dict(device="cpu", threads=1, batch=4, n_iter=1)
        with tempfile.TemporaryDirectory() as d:
            W = Path(d)
            np.save(W / "base.npy", np.zeros((19, 2, 2), np.float32))
            (W / "fake.py").write_text(fake)
            res = {}
            with mock.patch.dict(M.SCRIPTS, run=W / "fake.py"):
                for k, frames in ((1, None), (3, None), (2, None), (1, np.array([1, 4, 5, 9, 11, 17])),
                                  (2, np.array([1, 4, 5, 9, 11, 17]))):
                    sub = W / f"k{k}_{frames is None}"
                    sub.mkdir()
                    np.save(sub / "base.npy", np.zeros((19, 2, 2), np.float32))
                    f = M.run_stage(sub, "L", frames, q, k, lambda *_: None, {})
                    with np.load(f) as z:
                        res[(k, frames is None)] = (z["frames"], z["psi"], z["use"])
            for key in ((3, True), (2, True)):
                for i in range(3):
                    self.assertTrue(np.array_equal(res[key][i], res[(1, True)][i]), key)
            for i in range(3):
                self.assertTrue(np.array_equal(res[(2, False)][i], res[(1, False)][i]))


class TestFallbackRule(unittest.TestCase):
    """A failing fold (assertions or any error) keeps the parent config's maps byte for byte, and says so."""

    POST_OK = dict(veto_primary=dict(lcfs_found=1.0, inside_wall=0.9, ip_err_median=0.008, absZ_gt_0p3=4e-5,
                                     nonfinite_frames=0),
                   axis_veto_a67=dict(primary=dict(R_axis=dict(median_cm=-0.26, frac_gt3cm=0.0013),
                                                   Z_axis=dict(median_cm=-0.06, frac_gt3cm=0.025))))
    LI_OK = dict(coverage=0.997, frac_inrange_syn=1.0, frac_clamp=0.026, clip=dict(share_of_finite=0.019))

    def test_assertions(self):
        from fec.mast_nrsl import assertions
        q = _RF().NRSL_CASCADE
        self.assertEqual(assertions(q, self.POST_OK, self.LI_OK), [])
        cases = [("veto_primary", "lcfs_found", 0.98), ("veto_primary", "inside_wall", 0.84),
                 ("veto_primary", "ip_err_median", 0.031), ("veto_primary", "absZ_gt_0p3", 0.011),
                 ("veto_primary", "nonfinite_frames", 1)]
        for blk, k, v in cases:
            p = json.loads(json.dumps(self.POST_OK))
            p[blk][k] = v
            self.assertEqual(len(assertions(q, p, self.LI_OK)), 1, k)
        for ax, k, v in (("R_axis", "median_cm", 1.01), ("Z_axis", "median_cm", -1.2), ("R_axis", "frac_gt3cm", 0.051),
                         ("Z_axis", "median_cm", float("nan"))):
            p = json.loads(json.dumps(self.POST_OK))
            p["axis_veto_a67"]["primary"][ax][k] = v
            self.assertEqual(len(assertions(q, p, self.LI_OK)), 1, (ax, k, v))
        for k, v in (("coverage", 0.98), ("frac_inrange_syn", 0.94), ("clip", dict(share_of_finite=0.31))):
            self.assertEqual(len(assertions(q, self.POST_OK, dict(self.LI_OK, **{k: v}))), 1, k)
        # the frozen li tool's frac_clamp (0.625 / 1.665, not the calibration's bounds) does not gate
        self.assertEqual(assertions(q, self.POST_OK, dict(self.LI_OK, frac_clamp=0.9)), [])
        self.assertGreaterEqual(len(assertions(q, {}, {})), 5)

    def test_li_clip_uses_the_frozen_calibration(self):
        """The clip share is the share of finite li_cd outside the frozen model's [lo, hi] (fec.betap_li.predict_reg);
        its output bounds are the calibration's, not the 0.625 / 1.665 of the frozen tool's report."""
        from fec import mast_nrsl as M
        from fec.paths import WORK
        if not (WORK / "cdiff" / "frozen.json").exists():
            self.skipTest("frozen.json not installed")
        m = json.loads((WORK / "cdiff" / "frozen.json").read_text())["models"]["alone"]
        lo, hi = m["lo"][0], m["hi"][0]
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d) / "li.npz", li_cyl=np.array([lo - 0.1, lo, 1.0, hi, hi + 0.5, np.nan, 1.2, 1.3]))
            c = M.li_clip(Path(d))
        self.assertAlmostEqual(c["share_of_finite"], 2 / 7)
        self.assertAlmostEqual(c["below"], 1 / 7)
        self.assertAlmostEqual(c["above"], 1 / 7)
        self.assertAlmostEqual(c["output_bounds"][0], 0.6196, places=3)
        self.assertAlmostEqual(c["output_bounds"][1], 1.7819, places=3)

    # ---- v12.1 (amendment 68 addendum 2): fallback only for failing assertions; every other error stops the build
    def _setup(self, d, bad_pin=False):
        RF = _RF()
        maps = d / "maps"
        maps.mkdir()
        np.save(maps / "psi.npy", np.random.default_rng(0).standard_normal((5, 65, 65)).astype(np.float32))
        np.savez(maps / "meta.npz", psi_ax=np.arange(5.0))
        r = json.loads(json.dumps(RF.RECIPES["final_MTfull_mmk_nrsl"]))
        if bad_pin:
            r["mast"]["composite"]["nrsl"]["code_sha256"]["fec/nrsl_op.py"] = "0" * 64
        (d / "recipe.json").write_text(json.dumps(r))
        argv = ["run", "--config", "mast_private_test", "--recipe", str(d / "recipe.json"), "--maps", str(maps),
                "--passive", str(d / "I.npy"), "--out", str(d / "node"), "--workers", "1"]
        return maps, d / "node", argv

    def _main(self, M, argv):
        env0 = dict(os.environ)
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return M.main(argv)
        finally:
            os.environ.clear()
            os.environ.update(env0)

    def _fake_chain(self, M, W_holder, veto_lcfs):
        """Stand-ins for prep / the child processes that write the files the chain reads (5 frames); the post step's
        LCFS-found share is veto_lcfs."""
        def prep(W, *a, **k):
            W_holder.append(W)
            np.save(W / "base.npy", np.zeros((5, 65, 65), np.float32))
            return dict(N=5, base_sha256="b" * 64, maps_psi_sha256="m" * 64)

        def sh(cmd, log, env):
            s = " ".join(str(c) for c in cmd)
            W = W_holder[0]
            if "nrsl_li.py" in s:
                np.savez(W / "li.npz", li=np.ones(5), li_est_raw=np.ones(5), filled=np.zeros(5, bool), li_cyl=np.ones(5))
                (W / "li.json").write_text(json.dumps(dict(coverage=1.0, frac_inrange_syn=1.0, frac_clamp=0.0)))
            elif "nrsl_post.py" in s:
                np.savez(W / "casc_cpu.npz", psi=np.ones((5, 65, 65), np.float32), src=np.zeros(5, int))
                np.save(W / "primary_cpu.npy", np.full((5, 65, 65), 2.0, np.float32))
                p = json.loads(json.dumps(TestFallbackRule.POST_OK))
                p["veto_primary"]["lcfs_found"] = veto_lcfs
                (W / "post_cpu.json").write_text(json.dumps(p))

        def run_stage(W, st, frames, q, procs, log, env):
            f = W / f"stage_{st}_cpu.npz"
            np.savez(f, frames=np.arange(5), psi=np.ones((5, 65, 65), np.float32), use=np.ones(5, bool))
            return f
        return [mock.patch.object(M, "check_code", lambda q: []), mock.patch.object(M, "prep", prep),
                mock.patch.object(M, "sh", sh), mock.patch.object(M, "run_stage", run_stage),
                mock.patch.object(M, "li_clip", lambda W: dict(share_of_finite=0.0)),
                mock.patch.object(M, "write_meta", lambda p, o: np.savez(o, psi_ax=np.zeros(5)))]

    def test_code_check_error_stops(self):
        """A non-assertion error (here the code / settings check) exits non-zero, writes no node maps, and records the
        cause in nrsl.json (status error)."""
        from fec import mast_nrsl as M
        with tempfile.TemporaryDirectory() as d:
            maps, out, argv = self._setup(Path(d), bad_pin=True)
            self.assertEqual(self._main(M, argv), 1)
            self.assertFalse((out / "psi.npy").exists())
            self.assertFalse((out / "meta.npz").exists())
            j = json.loads((out / "nrsl.json").read_text())
            self.assertEqual(j["status"], "error")
            self.assertIsNone(j["fallback"])
            self.assertIn("nrsl_op.py", j["error"])

    def test_child_error_stops(self):
        """A child process's non-zero exit (here the li step) exits non-zero with no node maps, also after a stale
        node from an earlier run."""
        from fec import mast_nrsl as M
        with tempfile.TemporaryDirectory() as d:
            maps, out, argv = self._setup(Path(d))
            out.mkdir()
            (out / "psi.npy").write_bytes(b"stale")
            W = []
            pat = self._fake_chain(M, W, 1.0)

            def sh(cmd, log, env):
                raise RuntimeError("exit 3: nrsl_li.py")
            pat[2] = mock.patch.object(M, "sh", sh)
            for p in pat:
                p.start()
            try:
                self.assertEqual(self._main(M, argv), 1)
            finally:
                for p in pat:
                    p.stop()
            self.assertFalse((out / "psi.npy").exists())
            self.assertFalse((out / "meta.npz").exists())
            j = json.loads((out / "nrsl.json").read_text())
            self.assertEqual((j["status"], j["error"]), ("error", "RuntimeError: exit 3: nrsl_li.py"))

    def test_assertion_failure_keeps_parent_maps(self):
        """Failing label-free assertions: exit 0, the node holds the parent maps and meta byte for byte, nrsl.json
        status ok, fallback True with the reason."""
        from fec import mast_nrsl as M
        with tempfile.TemporaryDirectory() as d:
            maps, out, argv = self._setup(Path(d))
            W = []
            pat = self._fake_chain(M, W, 0.5)
            for p in pat:
                p.start()
            try:
                self.assertEqual(self._main(M, argv), 0)
            finally:
                for p in pat:
                    p.stop()
            self.assertEqual((out / "psi.npy").read_bytes(), (maps / "psi.npy").read_bytes())
            self.assertEqual((out / "meta.npz").read_bytes(), (maps / "meta.npz").read_bytes())
            j = json.loads((out / "nrsl.json").read_text())
            self.assertEqual((j["status"], j["fallback"]), ("ok", True))
            self.assertTrue(any("LCFS found" in s for s in j["reasons"]), j["reasons"])
            self.assertEqual(j["psi_sha256"], sha_file(maps / "psi.npy"))

    def test_assertion_pass_writes_cascade(self):
        from fec import mast_nrsl as M
        with tempfile.TemporaryDirectory() as d:
            maps, out, argv = self._setup(Path(d))
            W = []
            pat = self._fake_chain(M, W, 1.0)
            for p in pat:
                p.start()
            try:
                self.assertEqual(self._main(M, argv), 0)
            finally:
                for p in pat:
                    p.stop()
            self.assertEqual((out / "psi.npy").read_bytes(), (W[0] / "primary_cpu.npy").read_bytes())
            j = json.loads((out / "nrsl.json").read_text())
            self.assertEqual((j["status"], j["fallback"], j["assertions"]), ("ok", False, "pass"))

    def test_run_final_stage(self):
        """st_mast_nrsl: a non-zero exit raises StageError with the cause from nrsl.json (the build stops); a fold that
        kept the parent maps returns fallback 'parent_maps' and prints the banner; the build-summary banner reads the
        manifest; a pass returns fallback None and prints nothing."""
        RF = _RF()
        with tempfile.TemporaryDirectory() as d:
            c = RF.Ctx(_args(root=str(Path(d) / "root"), dry_run=False), json.loads(json.dumps(RF.RECIPES[NRSL[0]])))
            D = RF.composite_dirs(c)
            D["nrsl"].mkdir(parents=True)
            logs = []
            c.log = logs.append

            def fail(*x, **k):
                raise RF.StageError("exit code 1: mast_nrsl.py run ...")
            (D["nrsl"] / "nrsl.json").write_text(json.dumps(dict(status="error", fallback=None,
                                                                  error="RuntimeError: code / settings check failed")))
            with mock.patch.object(RF.Ctx, "sh", fail):
                with self.assertRaises(RF.StageError) as cm:
                    RF.st_mast_nrsl(c)
            self.assertIn("RuntimeError: code / settings check failed", str(cm.exception))
            self.assertIn("the build stops", str(cm.exception))
            (D["nrsl"] / "nrsl.json").write_text(json.dumps(dict(status="error", fallback=None, error="x")))
            with mock.patch.object(RF.Ctx, "sh", lambda *x, **k: None):
                with self.assertRaises(RF.StageError):
                    RF.st_mast_nrsl(c)          # exit 0 but no ok report / no node maps: still stops
            np.save(D["nrsl"] / "psi.npy", np.zeros(1))
            j = dict(status="ok", fallback=True, reasons=["LCFS found 0.5 < 0.99"], prep=dict(maps_psi_sha256="m"))
            (D["nrsl"] / "nrsl.json").write_text(json.dumps(j))
            with mock.patch.object(RF.Ctx, "sh", lambda *x, **k: None):
                info = RF.st_mast_nrsl(c)
            self.assertEqual(info["fallback"], "parent_maps")
            self.assertTrue(any("NRS-L CASCADE NOT APPLIED" in s for s in logs))
            self.assertTrue(any("LCFS found 0.5" in s for s in logs))
            c.L.manifest_dir.mkdir(parents=True, exist_ok=True)
            c.manifest_file("mast_nrsl").write_text(json.dumps(dict(status="ok", info=info)))
            logs.clear()
            self.assertTrue(RF.nrsl_banner(c))
            self.assertTrue(any("PARENT config's maps" in s for s in logs))
            j.update(fallback=False, reasons=[])
            (D["nrsl"] / "nrsl.json").write_text(json.dumps(j))
            logs.clear()
            with mock.patch.object(RF.Ctx, "sh", lambda *x, **k: None):
                info = RF.st_mast_nrsl(c)
            self.assertIsNone(info["fallback"])
            self.assertFalse(any("NOT APPLIED" in s for s in logs))
            c.manifest_file("mast_nrsl").write_text(json.dumps(dict(status="ok", info=info)))
            self.assertFalse(RF.nrsl_banner(c))


class TestIncrementSmoothing(unittest.TestCase):
    """psi = base + W30 (cascade - base) per shot (fec/nrsl_post.py): equals tsm30 of (raw + increment) when base =
    tsm30(raw) (linearity), leaves untouched shots bit-identical, and uses the shipped weights."""

    def test_linearity_and_untouched_shots(self):
        from fec.mast_tsm import gaussian_weights, shot_segments
        rng = np.random.default_rng(3)
        shot = np.repeat([0, 1, 2], [7, 5, 9])
        t = np.concatenate([np.sort(rng.uniform(0, 300, k)) for k in (7, 5, 9)])
        raw = rng.standard_normal((len(shot), 65, 65))
        tsm = np.zeros_like(raw)
        for s, e in zip(*shot_segments(shot)):
            tsm[s:e] = np.tensordot(gaussian_weights(t[s:e], 30.0), raw[s:e], axes=1)
        base = tsm.astype(np.float32)
        casc = base.copy()
        dl = (rng.standard_normal((len(shot), 65, 65)) * 1e-2).astype(np.float32)
        dl[7:12] = 0                                        # shot 1: every frame falls back
        casc = casc + dl
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.save(d / "base.npy", base)
            np.savez(d / "inputs.npz", shot=shot, t_ms=t, Ip=np.full(len(shot), 5e5), p2=np.zeros(len(shot)))
            for st in ("L", "P"):
                np.savez(d / f"stage_{st}_x.npz", frames=np.arange(len(shot)), psi=casc if st == "L" else base,
                         use=np.ones(len(shot), bool) if st == "L" else np.zeros(len(shot), bool))
            r = subprocess.run([sys.executable, "-B", str(REPO / "fec" / "nrsl_post.py"), "--fold-dir", str(d),
                                "--tag", "_x", "--skip-eval"], capture_output=True, text=True, cwd=str(REPO))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            prim = np.load(d / "primary_x.npy")
        self.assertEqual(prim.dtype, np.float32)
        self.assertTrue(np.array_equal(prim[7:12], base[7:12]))       # untouched shot: bit-identical
        want = np.zeros_like(raw)
        for s, e in zip(*shot_segments(shot)):
            want[s:e] = np.tensordot(gaussian_weights(t[s:e], 30.0), raw[s:e] + (casc[s:e].astype(np.float64)
                                                                               - base[s:e]), axes=1)
        # tsm30(raw + increment) == base + W30 increment up to the float32 rounding of base
        self.assertLess(np.abs(prim - want).max(), 1e-5 * np.abs(want).max())
        w = gaussian_weights(t[:7], 30.0)
        self.assertTrue(np.allclose(w.sum(1), 1.0))


def _flat(s):
    return " ".join(s.split())


def _doc(name):
    """The template in the research tree, the rendered file in the package."""
    p = REPO / "fec" / "final_pkg_tmpl" / name
    return (p if p.exists() else REPO / name).read_text(encoding="utf-8")


class TestShippedText(unittest.TestCase):
    """G8: the NRS-L text says what the stage does, discloses (i) chosen on a DIII-D VAL testbed with DIII-D labels,
    (ii) li target label-free on MAST, calibrated on DIII-D TRAIN only, (iii) no MAST reconstruction anywhere; it names
    no other competitor and claims no originality."""

    def parts(self):
        r = _doc("README.md").split("**NRS-L cascade (`final_MTfull_mmk_nrsl`", 1)[1].split("\n\n", 1)[0]
        p = _doc("PROVENANCE.md").split("* **NRS-L cascade (`final_MTfull_mmk_nrsl`", 1)[1].split("\n* **", 1)[0]
        return _flat(r), _flat(p)

    def test_no_originality_or_competitor(self):
        r, p = self.parts()
        for t in (r, p):
            for w in (" ours", "our design", "our operator", "our own", "novel", "we designed", "we developed",
                      "invented", "first to", "our ", "Our "):
                self.assertNotIn(w, t)
        try:
            from tests.test_fec_provenance_tmpl import TestV10MmkDocs as T  # research tree only
            names = T.NAMES
        except (ImportError, AttributeError):
            names = ()
        if (REPO / "tests" / "test_fec_provenance_tmpl.py").exists():
            self.assertTrue(names)
        for n in names:
            self.assertNotIn(n, _doc("README.md") + _doc("PROVENANCE.md"))

    def test_fallback_rule_text(self):
        """v12.1 (amendment 68 addendum 2 (a)): only failing assertions keep the parent maps; other errors stop."""
        r, p = self.parts()
        self.assertIn(_flat("If a check fails, the node holds the parent config's maps byte for byte, the stage manifest "
                            "records `fallback` `parent_maps` with the reason, and the run log and the build summary "
                            "print a banner. Any other error in the stage (a step that fails, a missing file, a child "
                            "process that exits non-zero) writes no maps and stops the build with the cause"), r)
        self.assertNotIn("any step errors", r)

    def test_final_submissions_slot_agnostic(self):
        t = _flat(_doc("PROVENANCE.md"))
        for s in ("Our Final submissions use three of `final_A22li_dec_mmk`, `final_MTfull_mmk`, `final_MTfull`, "
                  "`final_A22li_dec_mmk_nrsl` and `final_MTfull_mmk_nrsl`; the submission record names which.",
                  "All Final submissions start from the same r = 2 maps of the same shipped MAST network",
                  "The `_nrsl` configs add the NRS-L cascade, which was added during the Final phase and has no "
                  "Development-phase read. No Final-phase score was used to choose any MAST component."):
            self.assertIn(s, t)
        self.assertNotIn("All three Final submissions carry the same MAST configuration", t)

    def test_disclosures(self):
        r, p = self.parts()
        for s in ("*Chosen on DIII-D, with DIII-D labels.*", "read once on a DIII-D VAL testbed with DIII-D labels",
                  "both scored against the DIII-D EFIT labels", "ΔS_map +0.0107",
                  "*li target: label-free on MAST, calibrated on DIII-D TRAIN only.*",
                  "Its linear calibration was fitted on DIII-D TRAIN frames against DIII-D EFIT li",
                  "No MAST li or other MAST label enters it.", "*No MAST reconstruction anywhere.*",
                  "No MAST or MAST-U EFIT output, reconstruction or statistic", "no MAST score of any kind",
                  "were never used to choose anything", "No Development-phase read of it exists, and no Final-phase "
                  "score was used.", "15 variants, 12 single-stage and 3 cascades"):
            self.assertIn(_flat(s), p, s)
        for s in ("It trains nothing and reads no label.", "Only the MAST `psirz` changes",
                  "the node holds the parent config's maps byte for byte", "`fallback` `parent_maps`",
                  "ψ = base + W30 (cascade − base)", "`MKL_CBWR=AVX2`", "fixed 256-frame batches in fold order",
                  "calibration frozen on DIII-D TRAIN data"):
            self.assertIn(_flat(s), r, s)

    def test_budget_statement(self):
        rd = _flat(_doc("README.md"))
        self.assertIn("`final_MTfull_mmk_nrsl` and `final_A22li_dec_mmk_nrsl` train nothing new and add no GPU-h", rd)
        # v12.1 (rev_v12 wordings 2 and 3; amendment 68 addendum 2): measured package-environment wall time on both CPU
        # models, the ledger sentence for the _nrsl configs, the determinism results, the slot-agnostic Final text
        wall = ("Measured in this package's environment (Python 3.11, torch 2.13, 8 threads, 68,373-frame fold): stage L "
                "7,133–7,161 s on Xeon Gold 6342 pods and 9,014 s on an AMD EPYC 7543 pod; the whole stage "
                "8,179–8,208 s and 10,189 s, i.e. about 2.3–2.9 h of wall clock depending on the CPU model (earlier "
                "runs of the same stage code in torch 2.4.1 on Xeon: stage L 6,931–7,602 s).")
        self.assertIn(wall, rd)
        for s in ("about 2.3–2.9 h of wall clock per fold depending on the CPU model",
                  "the same wall-clock ledger with that stage's CPU time added gives about 26.5–27.1 PCIe-h for "
                  "`final_A22li_dec_mmk_nrsl` and about 25.8–26.4 for `final_MTfull_mmk_nrsl`",
                  "two pods with the same CPU model (Xeon Gold 6342) wrote bit-identical stage-L, cascade and "
                  "smoothed maps, and 1 versus 3 processes gave bit-identical maps on a 1,163-frame subset",
                  "an AMD EPYC 7543 pod differed from the Xeon pods on 14 of 68,373 frames by at most 4.0e-10 of the "
                  "ψ range (no stage choice changed): the stage is bit-identical on one CPU model, not across CPU "
                  "models"):
            self.assertIn(s, rd)
        for s in ("2–2.5 h", "stage L took 6,931–7,602 s per fold", "Python 3.11 / torch 2.4.1 environment"):
            self.assertNotIn(s, rd)
        b = REPO / "fec" / "final_pkg_tmpl" / "budget.md"
        bt = _flat(b.read_text(encoding="utf-8") if b.exists() else _doc("README.md"))
        self.assertIn("| CPU (0 GPU-h) |", bt)
        self.assertIn("CPU work at inference in float64, no GPU. " + wall, bt)
        self.assertIn("— / **22.83** (= `final_MTfull_mmk`; the cascade is CPU)", bt)


if __name__ == "__main__":
    unittest.main()
