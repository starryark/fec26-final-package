"""Final package (agent fpkg 2026-10-12): run_final Final presets, the MAST composite helpers,
the Final dataset routing and the starter issue #12 Thomson cases.  CPU only, synthetic data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class TestIssue12Thomson(unittest.TestCase):
    """Organiser (starter issue #12): the Final inputs contain one DIII-D Thomson edge-padding
    case and one MAST core out-of-order sample.  The extraction must neither crash nor change."""

    def setUp(self):
        from fec.extract import ts_interp
        self.ts = ts_interp
        self.t = np.arange(0.0, 400.0, 5.0)                    # EFIT frame times
        tt = np.arange(0.0, 400.0, 8.0)                         # Thomson sample times
        rng = np.random.default_rng(0)
        self.times = tt
        self.prof = (100 + 50 * rng.random((len(tt), 4))).astype(np.float64)

    def test_d3d_edge_padding_row_is_ignored(self):
        # train shots d3d_shot_17e313e867 / 29194d4b4f: one trailing row with time 0 and no data
        ref = self.ts(self.times, self.prof, self.t)
        tp = np.r_[self.times, 0.0]
        pp = np.vstack([self.prof, np.zeros((1, 4))])
        out = self.ts(tp, pp, self.t)
        np.testing.assert_array_equal(out, ref)

    def test_mast_core_out_of_order_sample_does_not_crash(self):
        k = 20
        tp = self.times.copy()
        tp[k], tp[k + 1] = tp[k + 1], tp[k]                     # one sample out of order
        pp = self.prof.copy()
        pp[[k, k + 1]] = pp[[k + 1, k]]                         # (time, value) pairs kept
        out = self.ts(tp, pp, self.t)
        ref = self.ts(self.times, self.prof, self.t)
        np.testing.assert_array_equal(out, ref)                 # sorting restores the series
        tq = self.times.copy()
        tq[k] = self.times[k - 3]                               # a genuinely misplaced stamp
        out2 = self.ts(tq, self.prof, self.t)
        self.assertEqual(out2.shape, ref.shape)
        self.assertTrue(np.isfinite(out2[(self.t > 10) & (self.t < 390)]).all())

    def test_duplicate_time_does_not_crash(self):
        tp = self.times.copy()
        tp[10] = tp[11]
        out = self.ts(tp, self.prof, self.t)
        self.assertTrue(np.isfinite(out[(self.t > 10) & (self.t < 390)]).all())

    def test_monotone_keep_is_the_organiser_rule(self):
        from fec.final_data import core_anomaly, monotone_keep
        t = np.array([0.0, 5.0, 10.0, 7.0, 15.0, 15.0, 20.0])
        np.testing.assert_array_equal(monotone_keep(t),
                                      [True, True, True, False, True, False, True])
        te = np.ones((7, 3))
        self.assertEqual(core_anomaly(t, te, te), (2, False))
        tp = np.r_[np.arange(5.0), 0.0]
        tep = np.vstack([np.ones((5, 2)), np.zeros((1, 2))])
        self.assertEqual(core_anomaly(tp, tep, tep), (1, True))


def _scan_row(stem, core_back=0, core_dup=0, edge_back=0):
    ax = lambda **k: dict(dict(n_dup=0, n_back=0, n_gap=0, nan=0), **k)  # noqa: E731
    return {"stem": stem, "efit": ax(), "mag": ax(), "ipt": ax(), "box_empty_mag": 0,
            "box_empty_ip": 0, "efit_after_ip_end": 0, "efit_after_mag_end": 0, "ch": {},
            "ts_core": {"axis": ax(n_back=core_back, n_dup=core_dup), "nch_vs_names": [40, 40]},
            "ts_edge": {"axis": ax(n_back=edge_back), "nch_vs_names": [10, 10]}}


class TestIssue12Guard(unittest.TestCase):
    def _guard(self, rows):
        from fec.input_fixes2 import guard
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "x_scan.jsonl", "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            return guard("x", out=d)

    def test_edge_padding_passes_input_guard(self):
        self.assertTrue(self._guard([_scan_row("a", edge_back=1), _scan_row("b")]))

    def test_core_issue12_tolerance(self):
        from fec.final_data import guard_conditions, tolerable_core_issue12
        rows = [_scan_row("a", core_back=1), _scan_row("b")]
        self.assertFalse(self._guard(rows))                    # the original guard stops
        self.assertEqual([k for k, v in guard_conditions(rows).items() if v],
                         ["Thomson core axis dup/back"])
        t = np.array([0.0, 5.0, 10.0, 7.0, 15.0])
        te = np.ones((5, 3))
        ok, det = tolerable_core_issue12(rows, lambda s: (t, te, te))
        self.assertTrue(ok, det)
        many = np.r_[np.arange(10.0), np.arange(5.0)]          # 5 non-monotone data rows
        ok, det = tolerable_core_issue12(rows, lambda s: (many, np.ones((15, 2)),
                                                          np.ones((15, 2))))
        self.assertFalse(ok)
        ok, det = tolerable_core_issue12(rows, None)            # cannot re-read: stop
        self.assertFalse(ok)
        bad = [_scan_row("a", core_back=1)]
        bad[0]["efit"]["n_back"] = 1                            # another condition fails too
        self.assertFalse(tolerable_core_issue12(bad, lambda s: (t, te, te))[0])

    def test_final_dataset_routing(self):
        import os
        from fec.final_data import FINAL_REPO, PUBLIC_REPO, dataset_repo, remote_files
        old = os.environ.pop("FEC_HF_DATASET", None)
        try:
            self.assertEqual(dataset_repo("mast_private_test"), FINAL_REPO)
            self.assertEqual(dataset_repo("private_test"), FINAL_REPO)
            self.assertEqual(dataset_repo("diii_d_public_test"), PUBLIC_REPO)
            self.assertEqual(FINAL_REPO, "ganatma/fusion-equilibrium-challenge-final")   # organisers' Final guide 2026-10-19
        finally:
            if old is not None:
                os.environ["FEC_HF_DATASET"] = old
        files = ["data/mast_private_test/b.parquet", "data/mast_private_test/a.parquet",
                 "README.md", "diii_d_private_test/x.parquet"]
        self.assertEqual(remote_files("r", "mast_private_test", files),
                         (["data/mast_private_test/a.parquet", "data/mast_private_test/b.parquet"],
                          "data/mast_private_test/"))
        self.assertEqual(remote_files("r", "diii_d_private_test", files)[1], "diii_d_private_test/")


class TestFill3(unittest.TestCase):
    def test_fill3_rule(self):
        from fec.final_mast import postprocess_fill3
        shot = np.array([0] * 6 + [1] * 5 + [2] * 4)
        x = np.array([1.0, 1.2, np.nan, 1.1, 1.3, 1.0,          # shot 0: 5 finite -> median
                      7.0, np.nan, np.nan, np.nan, np.nan,      # shot 1: 1 finite -> global
                      2.0, 2.2, np.nan, 2.1])
        y = postprocess_fill3(x, shot)
        glob = np.nanmedian(x)
        self.assertTrue(np.isfinite(y).all())
        self.assertAlmostEqual(np.median(y[7:11]), min(max(glob, 0.0), 8.0))
        self.assertTrue((y >= 0).all() and (y <= 8).all())
        # min_finite=1 is the production (pre-fill3) rule: the lone value is copied
        y1 = postprocess_fill3(x, shot, min_finite=1)
        self.assertTrue(np.allclose(y1[7:11], 7.0))


class TestSmoothPsi(unittest.TestCase):
    def test_equals_mast_tsm_psi(self):
        from fec.final_mast import smooth_psi
        from fec.mast_tsm import smooth_maps
        rng = np.random.default_rng(1)
        shot = np.repeat(np.arange(4), [7, 3, 9, 5])
        t = np.concatenate([np.sort(rng.uniform(0, 300, n)) for n in (7, 3, 9, 5)])
        psi = rng.standard_normal((len(shot), 65, 65)).astype(np.float32)
        fake = lambda x: {k: np.zeros(len(x)) for k in  # noqa: E731
                          ("psi_ax", "psi_b", "Rax", "Zax", "limited")}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "src").mkdir()
            np.save(d / "src" / "psi.npy", psi)
            for f in ("scalars.npz", "betan_records.npz"):
                np.savez(d / "src" / f, a=np.zeros(1))
            np.save(d / "src" / "betan_v2.npy", np.zeros(1))
            (d / "src" / "betan_v2.json").write_text("{}")
            smooth_maps(d / "src", d / "a", shot, t, 30.0, meta_fn=fake, batch=8, log=lambda m: 0)
            smooth_psi(d / "src", d / "b", shot, t, 30.0, batch=8, log=lambda m: 0, meta_fn=fake)
            self.assertEqual((d / "a" / "psi.npy").read_bytes(), (d / "b" / "psi.npy").read_bytes())


class TestPlanAvgF05(unittest.TestCase):
    def test_plan_matches_cand_avg_f05(self):
        from fec import mast_pipeline as MP
        from fec.final_mast import AVG_F05_MEMBERS, plan_avg_f05
        with tempfile.TemporaryDirectory() as d:
            plan = MP.Plan(d, "mast_private_test", 1000, 1000)
            plan_avg_f05(plan)
            keys = [st.key for st in plan.steps]
            self.assertEqual(keys[:4], [("zsc_c1", "stage1"), ("zsc_nopas", "stage1"),
                                        ("zsc_c1_lcc", "stage1"), ("zfix0_c1", "stage1")])
            self.assertIn(("dc", "merge"), keys)
            self.assertIn(("zsc_c1_tsm30", "tsm"), keys)
            self.assertEqual(keys[-2:], [("avg_f05", "stage2"), ("avg_f05", "betan")])
            avg = plan.by_key[("avg_f05", "average")]
            self.assertEqual(avg.params["weights"], [0.2] * 5)
            members = [Path(x).name for x in avg.cmd[avg.cmd.index("--member") + 1::2][:5]]
            self.assertEqual(tuple(members), AVG_F05_MEMBERS)
            lcc = plan.by_key[("zsc_c1_lcc", "stage1")]
            self.assertIn("--limiter-cc", lcc.cmd)
            zf = plan.by_key[("zfix0_c1", "stage1")]
            self.assertEqual(zf.cmd[zf.cmd.index("--zfix") + 1], "0.0")
            self.assertNotIn("--sol-frac", plan.by_key[("zsc_c1", "stage1")].cmd)


def _args(**kw):
    base = dict(split="public_test", subset_shots=0, workers=4, owner="t", gpu_priority="normal",
                force=False, dry_run=True, no_runlog=True, remote=False, loff_dir=None,
                lon_dir=None, mast_dir=None, blim="recipe", d3d_dir=None, q95h_src=None,
                submission_name="submission", allow_unreviewed=False, root=None, adopt=None,
                verify_only="", ref_d3d=None, ref_mast=None, ref_submission=None, d3d_work=None,
                psi_dtype=None, no_model_hash=True, with_train=False, d3d_trainer="recipe",
                with_upload=False, confirm_upload=False, hf_repo=None, hf_revision=None,
                pointer_out=None, pointer_offline=False, amort_net_dir=None,
                mast_scalars_dir=None, mast_jobs=1, stage="all")
    base.update(kw)
    return SimpleNamespace(**base)


class TestRunFinalPresets(unittest.TestCase):
    def test_presets_and_stage_order(self):
        import fec.run_final as RF
        for name, b, rec in (("final_A", "d3d_ip_C2_w3k44", "fec/recipes_final/final_ip_C2_w3k44.json"),
                             ("final_B", "d3d_ip_C2_w3e44", "fec/recipes_b/final_ip_C2_w3e44.json"),
                             ("final_A22li", "d3d_ip_C2_w3k22liA",
                              "fec/recipes_final/final_ip_C2_w3k22liA.json")):
            r = RF.RECIPES[name]
            self.assertEqual(r["d3d"]["b"]["name"], b)
            self.assertEqual(r["d3d"]["b"]["recipe"], rec)
            self.assertTrue((REPO / rec).is_file())
            self.assertEqual(r["d3d"]["q95h"]["mode"], "G1")
            self.assertEqual(r["d3d"]["lirep4"]["target"], 4.9)
            self.assertEqual(r["psi_dtype"], "float32")
            self.assertEqual(r["d3d"]["ip_timebase"], "idx")
            self.assertEqual(r["mast"]["composite"]["maps"]["sol_frac"], 0.55)
            self.assertEqual(r["mast"]["composite"]["maps"]["net"], "n7h_f0.55")
            st = RF.all_stages(r)
            self.assertEqual(st, ["download", "ipguard", "extract", "order", "dataset", "passive",
                                  "mast_amort", "mast_physics", "d3d_b", "d3d_q95h",
                                  "d3d_lirep4", "mast_composite", "submission"])
            RF.load_recipe(name, "C:/nonexistent_root_for_test")
        # recipes without lirep4/composite keep their stage lists
        self.assertNotIn("d3d_lirep4", RF.all_stages(RF.RECIPES["sub07"]))
        self.assertIn("mast_loff", RF.all_stages(RF.RECIPES["sub07"]))

    def test_w3k44_recipe_members(self):
        r = json.loads((REPO / "fec/recipes_final/final_ip_C2_w3k44.json").read_text())
        e = json.loads((REPO / "fec/recipes_b/final_ip_C2_w3e44.json").read_text())
        self.assertEqual(len(r["members"]), 44)
        self.assertEqual(sum("bxw3e" in m["tag"] for m in r["members"]), 44)
        self.assertEqual(sum("bxw3e" in m["tag"] for m in e["members"]), 16)
        self.assertEqual(r["fallback"], e["fallback"])            # OLD 28 notsL fallback
        self.assertAlmostEqual(sum(m["weight"] for m in r["members"]), 1.0)

    def test_composite_validation(self):
        import fec.run_final as RF
        bad = json.loads(json.dumps(RF.RECIPES["final_A"]))
        bad["mast"]["composite"]["betan_fill"]["rule"] = "production"
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "r.json"
            f.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                RF.load_recipe(str(f), d)

    def test_dirs_and_submit_source(self):
        import fec.run_final as RF
        with tempfile.TemporaryDirectory() as d:
            c = RF.Ctx(_args(root=d), json.loads(json.dumps(RF.RECIPES["final_A"])))
            D = RF.composite_dirs(c)
            self.assertEqual(D["tsm"].name, "amort_n7h_f0.55_tsm30")
            self.assertEqual(D["passive"].name, "passive_f0.55")
            self.assertEqual(RF._mast_input(c, "final"), D["final"])
            src, out = RF.lirep4_dirs(c)
            self.assertEqual(src.name, "d3d_ip_C2_w3k44_q95h")
            self.assertEqual(out.name, "d3d_ip_C2_w3k44_q95h_lirep4")
            self.assertEqual(RF.d3d_submit_dir(c)[0], out)        # dry run: the lirep4 dir
            self.assertEqual(RF.composite_scalars_dir(c), D["physics"] / "avg_f05")
            c2 = RF.Ctx(_args(root=d, mast_scalars_dir="X:/stored"),
                        json.loads(json.dumps(RF.RECIPES["final_A"])))
            self.assertEqual(RF.composite_scalars_dir(c2), Path("X:/stored"))


if __name__ == "__main__":
    unittest.main()


class TestSyntheticInstall(unittest.TestCase):
    """train.py --synthetic: all-or-nothing install of the SHA-256-pinned synthetic chunks."""
    def _train(self):
        import importlib.util
        for d in (REPO, REPO / "fec" / "final_pkg_tmpl"):
            if (d / "train.py").exists() and (d / "infer.py").exists():
                spec = importlib.util.spec_from_file_location("pkg_train_under_test", d / "train.py")
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                return m
        self.skipTest("train.py not found")

    def test_install_checks_every_hash(self):
        import hashlib, tempfile
        T = self._train()
        with tempfile.TemporaryDirectory() as d:
            src, gen = Path(d) / "src", Path(d) / "gen"
            src.mkdir()
            files = []
            for i in range(3):
                b = f"chunk {i}".encode()
                (src / f"chunk_l0_{i:04d}.npz").write_bytes(b)
                files.append(dict(path=f"synthetic/gen_v7h_f0.55/chunk_l0_{i:04d}.npz",
                                  sha256=hashlib.sha256(b).hexdigest()))
            dman = {"synthetic": {"files": files}}
            self.assertEqual(T.install_synthetic(src, gen, dman), 3)
            self.assertEqual(sorted(p.name for p in gen.iterdir()),
                             [f"chunk_l0_{i:04d}.npz" for i in range(3)])
            (src / "chunk_l0_0001.npz").write_bytes(b"tampered")
            gen2 = Path(d) / "gen2"
            with self.assertRaises(SystemExit):
                T.install_synthetic(src, gen2, dman)
            self.assertFalse(gen2.exists())
            (src / "chunk_l0_0002.npz").unlink()
            with self.assertRaises(SystemExit):
                T.install_synthetic(src, gen2, dman)


@unittest.skipUnless((REPO / "fec" / "final_pkg.py").exists(),
                     "builder tests run in the research repo only")
class TestPackageBuilder(unittest.TestCase):
    def test_job_line_and_scan(self):
        from fec.final_pkg import FORBIDDEN, job_line
        ln = ("OMP_NUM_THREADS=10 python -B fec/train_d3d_b.py --tag t --alldata --seed 3 "
              "--vram_gb 6 > /x/logs/t.log 2>&1")
        self.assertEqual(job_line(ln), "python -B fec/train_d3d_b.py --tag t --alldata --seed 3 "
                                       "--vram_gb 6")
        for bad in ("C:" + "/Users/someone/x", "hf_" + "a" * 34, "/work" + "space/fec_work"):
            self.assertTrue(FORBIDDEN.search(bad), bad)
        self.assertIsNone(FORBIDDEN.search("fec_work/models/d3d_bxw3e_shape_s0/model.pt"))

    def test_closure_covers_final_entrypoints(self):
        from fec.final_pkg import closure
        mods = set(closure())
        for m in ("run_final", "final_mast", "final_build", "final_data", "mast_pipeline",
                  "mast_zsc", "mast_v1", "betan_transfer", "q95_hybrid", "d3d_rdspec",
                  "predict_d3d_b", "train_d3d_bx", "amort_gen7", "amort_net2", "paths"):
            self.assertIn(m, mods)

    def test_frozen_ranges_match_reference_build(self):
        from fec.final_build import load_ranges
        r = load_ranges()
        self.assertEqual(r["q95"], (0.2932361364364624, 19.520709991455078))
        self.assertEqual(r["betaN"], (0.0, 9.815239906311035))


class TestW3k24Subset(unittest.TestCase):
    def test_subset_of_w3k44_with_balanced_families(self):
        a = json.loads((REPO / "fec/recipes_final/final_ip_C2_w3k24.json").read_text())
        k = json.loads((REPO / "fec/recipes_final/final_ip_C2_w3k44.json").read_text())
        tags = [m["tag"] for m in a["members"]]
        self.assertEqual(len(tags), 24)
        self.assertTrue(set(tags) <= {m["tag"] for m in k["members"]})
        self.assertEqual(sum("notsL" in t for t in tags), 12)
        self.assertAlmostEqual(sum(m["weight"] for m in a["members"] if "notsL" in m["tag"]), 0.5)
        self.assertEqual(a["fallback"], k["fallback"])


class TestRunningMaxThomsonRule(unittest.TestCase):
    """Organiser rule (issue #12, 2026-10-11): keep a sample only if later than EVERY earlier
    sample; no t <= 0 filter and no truncation at the first backward step."""

    def test_keep_mask(self):
        from fec.extract_final import running_max_keep
        t = np.array([0.0, 10.0, 20.0, 15.0, 30.0, 30.0, 40.0, np.nan, 50.0, -5.0, 60.0])
        np.testing.assert_array_equal(
            running_max_keep(t),
            [True, True, True, False, True, False, True, False, True, False, True])
        self.assertTrue(running_max_keep(np.array([-3.0, -1.0, 0.0]))[0])   # t <= 0 is kept

    def test_interp_identical_on_public_cases_and_drops_misplaced_sample(self):
        from fec.extract import ts_interp
        from fec.extract_final import ts_interp_runmax
        t = np.arange(0.0, 400.0, 5.0)
        tt = np.arange(0.0, 400.0, 8.0)
        prof = 100 + 50 * np.random.default_rng(0).random((len(tt), 4))
        np.testing.assert_array_equal(ts_interp_runmax(tt, prof, t), ts_interp(tt, prof, t))
        tp = np.r_[tt, 0.0, 0.0]                                  # DIII-D edge padding rows
        pp = np.vstack([prof, np.zeros((2, 4))])
        np.testing.assert_array_equal(ts_interp_runmax(tp, pp, t), ts_interp(tp, pp, t))
        tq = tt.copy()
        tq[20] = tt[17]                                           # a misplaced, data-carrying sample
        a, b = ts_interp(tq, prof, t), ts_interp_runmax(tq, prof, t)
        self.assertFalse(np.array_equal(a, b, equal_nan=True))    # sort keeps it, runmax drops it
        keep = np.ones(len(tt), bool); keep[20] = False
        np.testing.assert_array_equal(b, ts_interp(tq[keep], prof[keep], t))
        self.assertTrue(np.isfinite(b[(t > 10) & (t < 390)]).all())   # later samples still used

    def test_patch_and_run_final_default(self):
        import fec.extract as EX
        import fec.extract_final as XF
        import fec.run_final as RF
        orig = EX.ts_interp
        try:
            XF._patch()
            self.assertIs(EX.ts_interp, XF.ts_interp_runmax)
        finally:
            EX.ts_interp = orig
        with tempfile.TemporaryDirectory() as d:
            rec = json.loads(json.dumps(RF.RECIPES["final_A"]))
            self.assertEqual(RF.ts_rule(RF.Ctx(_args(root=d, split="private_test"), rec)), "runmax")
            self.assertEqual(RF.ts_rule(RF.Ctx(_args(root=d), rec)), "sort")
            self.assertEqual(RF.ts_rule(RF.Ctx(_args(root=d, split="private_test", ts_rule="sort"),
                                               rec)), "sort")


class TestFinalA22li(unittest.TestCase):
    """final_A22li = W3k22 + li2ft arm A (docs D3D_LI2FT22_20261013 section 7).  Runs inside the package (its
    configs/) and in the research repo (train spec written by fec/final_pkg.py train_configs into a temporary
    package skeleton with the train.py/infer.py templates)."""
    F_SUFFIX = ["shape_s0", "shape_s1", "shapeA_s2", "shapeL_s3", "shape_s40", "shape_s41", "shapeA_s42",
                "shapeL_s43", "shape_s80", "shape_s81", "shapeA_s82"]
    N_SUFFIX = ["notsL_s5", "notsL_s6", "notsL_s7", "notsL_s45", "notsL_s46", "notsL_s47", "notsL_s85",
                "notsL_s86", "notsL_s87", "notsL_s125", "notsL_s126"]
    GROUPS = ["w3k22_F1", "w3k22_F2a", "w3k22_F2b", "old_notsL_axij", "old_notsL_axkl", "li2ftA22"]
    _tmp = None

    @classmethod
    def setUpClass(cls):
        if (REPO / "configs" / "train_d3d_final_A22li.json").exists():          # inside the package
            cls.pkg = REPO
            return
        if not (REPO / "fec" / "final_pkg.py").exists():
            raise unittest.SkipTest("neither package configs nor the package builder")
        import shutil
        from fec import final_pkg as FP
        missing = [str(src) for src, *_ in FP.TRAIN_GROUPS.values() if not Path(src).exists()]
        if missing:
            raise unittest.SkipTest(f"source job files absent: {missing[:2]}")
        cls._tmp = tempfile.TemporaryDirectory()
        pkg = Path(cls._tmp.name)
        FP.train_configs(pkg)
        for f in ("train.py", "infer.py"):
            shutil.copyfile(REPO / "fec" / "final_pkg_tmpl" / f, pkg / f)
        for cand in FP.CAND_GROUPS:
            (pkg / "configs" / f"{cand}.json").write_text(json.dumps(FP.recipe_for_package(cand)))
        (pkg / "data_manifest.json").write_text(json.dumps({"frozen": {}, "synthetic": {"files": []}}))
        (pkg / "weights_manifest.json").write_text(json.dumps({"files": []}))
        cls.pkg = pkg

    @classmethod
    def tearDownClass(cls):
        if cls._tmp is not None:
            cls._tmp.cleanup()

    def _recipe(self, name="final_ip_C2_w3k22liA.json"):
        return json.loads((REPO / "fec" / "recipes_final" / name).read_text())

    def _jobs(self, group):
        import shlex
        return [shlex.split(ln) for ln in
                json.loads((self.pkg / "configs" / "jobs" / f"{group}.json").read_text())]

    @staticmethod
    def _opt(toks, k):
        return toks[toks.index(k) + 1] if k in toks else None

    def test_members_and_fallback(self):
        r, a24 = self._recipe(), self._recipe("final_ip_C2_w3k24.json")
        self.assertEqual(sorted(m["tag"] for m in r["members"]),
                         sorted("d3d_li2ftA22_" + s for s in self.F_SUFFIX + self.N_SUFFIX))
        for m in r["members"]:
            self.assertEqual(m.get("kind", "b"), "b")
            self.assertAlmostEqual(m["weight"], 1 / 22)
        self.assertEqual(r["fallback"], a24["fallback"])            # the old-28 notsL, not fine-tuned
        self.assertEqual(len(r["fallback"]), 28)
        w3k24 = {m["tag"].replace("d3d_bxw3e_", "") for m in a24["members"]}
        self.assertEqual(w3k24 - set(self.F_SUFFIX + self.N_SUFFIX), {"shapeL_s83", "notsL_s127"})
        self.assertTrue(set(self.F_SUFFIX + self.N_SUFFIX) < w3k24)

    def test_config_is_a24_chain_with_new_members(self):
        cfg = json.loads((self.pkg / "configs" / "final_A22li.json").read_text())
        a24 = json.loads((self.pkg / "configs" / "final_A24.json").read_text())
        self.assertEqual(cfg["d3d"]["b"]["recipe"], "fec/recipes_final/final_ip_C2_w3k22liA.json")
        self.assertEqual(cfg["d3d"]["b"]["name"], "d3d_ip_C2_w3k22liA")
        self.assertNotIn("reference", cfg)

        def strip(c):
            c = json.loads(json.dumps(c))
            c["name"] = None
            c["d3d"]["b"].update(name=None, recipe=None)
            return c
        self.assertEqual(strip(cfg), strip(a24))

    def test_train_spec(self):
        spec = json.loads((self.pkg / "configs" / "train_d3d_final_A22li.json").read_text())
        self.assertEqual([g["name"] for g in spec["groups"]], self.GROUPS)       # fine-tune LAST
        self.assertEqual([g["n"] for g in spec["groups"]], [11, 6, 5, 14, 14, 22])
        self.assertEqual([g.get("li2_cache") for g in spec["groups"]], [None] * 5 + [True])
        for other in ("final_A24", "final_B", "final_C"):
            s = json.loads((self.pkg / "configs" / f"train_d3d_{other}.json").read_text())
            self.assertFalse(any("li2_cache" in g for g in s["groups"]), other)

    def test_finetune_jobs_start_from_the_bases_trained_before(self):
        from fec.train_d3d_bx import FT_SAME_AS_BASE, member_get_args

        def parse(toks):         # the member arguments, parsed as fec/train_d3d_bx.py parses them
            return member_get_args(toks[next(i for i, x in enumerate(toks) if x.endswith("train_d3d_b.py")) + 1:])
        base = {}
        for g in self.GROUPS[:3]:
            for t in self._jobs(g):
                base[self._opt(t, "--tag")] = t
        self.assertEqual(sorted(base), sorted("d3d_bxw3e_" + s for s in self.F_SUFFIX + self.N_SUFFIX))
        ft = self._jobs("li2ftA22")
        self.assertEqual({self._opt(t, "--tag") for t in ft}, {m["tag"] for m in self._recipe()["members"]})
        self.assertEqual(sorted(self._opt(t, "--ft_from") for t in ft), sorted(base))
        arm_a = {"--li2_rho": "0.1", "--li2_w": "4", "--li2_thr": "1.7", "--ft_lr_frac": "0.1", "--epochs": "12"}
        for t in ft:
            tag, src = self._opt(t, "--tag"), self._opt(t, "--ft_from")
            self.assertEqual(tag.replace("d3d_li2ftA22_", ""), src.replace("d3d_bxw3e_", ""))
            self.assertEqual(int(self._opt(t, "--seed")), 7000 + int(self._opt(base[src], "--seed")))
            self.assertIn("--alldata", t)
            self.assertIn("--alldata", base[src])
            for k, v in arm_a.items():
                self.assertEqual(self._opt(t, k), v, (tag, k))
            m, b = parse(t), parse(base[src])
            for k in ("feat",) + tuple(FT_SAME_AS_BASE):        # the base loss and net are unchanged
                self.assertEqual(getattr(m, k), getattr(b, k), (tag, k))
            self.assertEqual((m.ft_from, m.li2_rho, m.li2_w, m.li2_thr, m.ft_lr_frac, m.epochs),
                             (src, 0.1, 4.0, 1.7, 0.1, 12))

    def _dry(self, cfg):
        import subprocess
        with tempfile.TemporaryDirectory() as w:
            p = subprocess.run([sys.executable, "-B", str(self.pkg / "train.py"), "--config", cfg, "--stage", "all",
                                "--dry-run", "--work", w], cwd=str(self.pkg), capture_output=True, text=True,
                               timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        return [ln[2:] for ln in p.stdout.splitlines() if ln.startswith("$ ")]

    def test_train_dry_run_li2cache_before_d3d(self):
        cmds = self._dry("final_A22li")
        li2 = [i for i, c in enumerate(cmds) if "li2_torch.py" in c]
        bx = [i for i, c in enumerate(cmds) if "train_d3d_bx.py" in c]
        data = [i for i, c in enumerate(cmds) if "bnd_b.py" in c or "final_build.py" in c]
        self.assertEqual(len(li2), 2)                                 # build + check
        self.assertEqual(len(bx), 6)
        self.assertTrue(data and max(data) < min(li2) and max(li2) < min(bx))
        cache = str(Path("fec_work_idx") / "li2" / "cache_all")
        self.assertIn(" build ", cmds[li2[0]])
        self.assertIn("--splits train,val,test", cmds[li2[0]])
        self.assertIn(cache, cmds[li2[0]])
        self.assertIn(" check ", cmds[li2[1]])
        self.assertEqual(["--li2_cache" in cmds[i] for i in bx], [False] * 5 + [True])
        self.assertIn("li2ftA22.json", cmds[bx[-1]])
        self.assertIn(cache, cmds[bx[-1]])
        for other in ("final_A24", "final_B"):                         # unchanged: no li2 stage, no flag
            c2 = self._dry(other)
            self.assertFalse(any("li2_torch.py" in c or "--li2_cache" in c for c in c2), other)

    def test_weights_manifest_lists_the_finetuned_members(self):
        f = REPO / "weights_manifest.json"
        if not f.exists():
            self.skipTest("weights_manifest.json is written by the package build")
        rows = {r["tag"]: r for r in json.loads(f.read_text())["files"] if r.get("group") == "d3d"}
        r = self._recipe()
        for m in r["members"] + r["fallback"]:
            self.assertIn(m["tag"], rows)
        li = sorted(t for t in rows if t.startswith("d3d_li2ftA22_"))
        self.assertEqual(len(li), 22)
        for t in li:
            self.assertEqual(rows[t]["path"], f"d3d/{t}/model.pt")
            self.assertEqual(rows[t]["dest"], f"fec_work/models/{t}/model.pt")
            self.assertRegex(rows[t]["sha256"] or "", r"^[0-9a-f]{64}$")
            self.assertGreater(rows[t]["bytes"], 0)
