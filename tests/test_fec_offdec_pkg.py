"""final_A22li_dec = final_A22li + the off-span decoder stage (H1dec; agent decpkg 2026-10-14; EVIDENCE B61; family doc
amendment 26; research/reviews/REVIEW_H1DEC_20261014.md section 8).

Shipped in the Final package (tests/) and run in the research repo (a temporary package skeleton written by
fec/final_pkg.py).  CPU only, synthetic data only:
  * config = final_A22li + d3d.dec; same train spec; train.py order d3d -> offdec -> mast; offdec a no-op elsewhere;
  * run_final order q95h -> dec -> lirep4, lirep4 reads the decoded preds (its source on the dec fallback);
  * export / install_weights carry the 3 decoders for this config only; per-config disk peak;
  * fec/offdec.py: the stage refuses members whose bases differ by more than 1e-6; the correction has no in-span
    component (< 1e-8 RMS) and no NaN; q95 / betaN files are copied byte for byte; one seed trains bit-identically
    twice on one machine (CPU).
"""
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CFG = "final_A22li_dec"


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(m)
    return m


class _Pkg(unittest.TestCase):
    """The package dir: REPO itself inside the package, else a skeleton (configs, train/infer templates)."""
    _tmp = None

    @classmethod
    def setUpClass(cls):
        if (REPO / "configs" / f"{CFG}.json").exists():          # inside the package
            cls.pkg = REPO
            return
        if not (REPO / "fec" / "final_pkg.py").exists():
            raise unittest.SkipTest("neither package configs nor the package builder")
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
        for rel in FP.FEC_DATA:                                    # the members recipes infer.recipe_tags reads
            if rel.startswith("recipes"):
                (pkg / "fec" / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / "fec" / rel, pkg / "fec" / rel)
        (pkg / "data_manifest.json").write_text(json.dumps({"frozen": {}, "synthetic": {"files": []}}))
        (pkg / "weights_manifest.json").write_text(json.dumps({"files": []}))
        cls.pkg = pkg

    @classmethod
    def tearDownClass(cls):
        if cls._tmp is not None:
            cls._tmp.cleanup()

    def cfg(self, name=CFG):
        return json.loads((self.pkg / "configs" / f"{name}.json").read_text())

    def _dry(self, cfg, stage="all"):
        with tempfile.TemporaryDirectory() as w:
            p = subprocess.run([sys.executable, "-B", str(self.pkg / "train.py"), "--config", cfg, "--stage", stage,
                                "--dry-run", "--work", w], cwd=str(self.pkg), capture_output=True, text=True,
                               timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        return p.stdout, [ln[2:] for ln in p.stdout.splitlines() if ln.startswith("$ ")]


class TestDecConfig(_Pkg):
    def test_config_is_a22li_r2_plus_dec(self):
        dec, base, a22 = self.cfg(), self.cfg("final_A22li_r2"), self.cfg("final_A22li")
        q = dec["d3d"].pop("dec")
        self.assertNotIn("dec", base["d3d"])
        self.assertIn("_dec_lirep4", dec.pop("name"))
        base.pop("name")
        self.assertEqual(dec, base)                       # everything else is final_A22li_r2's
        self.assertEqual(dec["d3d"], a22["d3d"])          # DIII-D half = final_A22li's (+ dec)
        self.assertEqual(dec["mast"]["composite"]["maps"]["rscale"], 2)    # MAST half = r = 2 (EVIDENCE C66)
        self.assertEqual(q["seeds"], [40, 41, 42])
        self.assertEqual(q["fallback"], "src")
        self.assertEqual(q["suffix"], "_dec")
        self.assertEqual(q["dir"], "offdec/d3d_ip_C2_w3k22liA")
        self.assertEqual(q["tag"], "offdec_d3d_ip_C2_w3k22liA")
        self.assertEqual(q["chunk"], 8192)
        self.assertEqual(q["inspan_rms_max"], 1e-8)
        self.assertRegex(q["shipped_basis_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(q["ratio_band"], [0.5, 2.0])

    def test_train_spec_is_a22lis(self):
        a = json.loads((self.pkg / "configs" / f"train_d3d_{CFG}.json").read_text())
        b = json.loads((self.pkg / "configs" / "train_d3d_final_A22li.json").read_text())
        self.assertEqual(a["groups"], b["groups"])

    def test_train_order_d3d_offdec_mast(self):
        _, cmds = self._dry(CFG)
        bx = [i for i, c in enumerate(cmds) if "train_d3d_bx.py" in c]
        od = [i for i, c in enumerate(cmds) if "offdec.py" in c]
        mast = [i for i, c in enumerate(cmds) if "final_mast.py" in c or "amort_net2.py" in c]
        self.assertEqual(len(bx), 6)
        self.assertEqual([cmds[i].split("offdec.py ")[1].split()[0] for i in od],
                         ["basis", "envelope", "data", "train", "train", "train"])
        self.assertEqual([c.split("--seed ")[1].split()[0] for c in (cmds[i] for i in od[3:])], ["40", "41", "42"])
        self.assertTrue(max(bx) < min(od) and max(od) < min(mast))
        self.assertTrue(all("--device cuda" in cmds[i] for i in od[3:]))
        self.assertIn("final_ip_C2_w3k22liA.json", cmds[od[0]])
        sep = "\\" if "\\" in cmds[od[3]] else "/"
        self.assertIn(sep.join(["offdec", "d3d_ip_C2_w3k22liA", "dec_s40.pt"]), cmds[od[3]])
        self.assertIn(sep.join(["offdec_cache", "d3d_ip_C2_w3k22liA"]), cmds[od[2]])
        # --stage d3d trains the decoders right after the members (reproduce.sh full runs data, d3d, mast, export
        # unchanged); --stage offdec alone re-runs only them; the whole chain runs them once
        _, c2 = self._dry(CFG, "d3d")
        bx2 = [i for i, c in enumerate(c2) if "train_d3d_bx.py" in c]
        od2 = [i for i, c in enumerate(c2) if "offdec.py" in c]
        self.assertEqual((len(bx2), len(od2)), (6, 6))
        self.assertTrue(max(bx2) < min(od2))
        _, c3 = self._dry(CFG, "offdec")
        self.assertEqual(len([c for c in c3 if "offdec.py" in c]), 6)
        self.assertFalse(any("train_d3d_bx.py" in c for c in c3))

    def test_offdec_is_a_noop_for_other_configs(self):
        for other in ("final_A22li", "final_B", "final_A24"):
            if not (self.pkg / "configs" / f"{other}.json").exists():
                continue
            for st in ("all", "d3d"):
                _, cmds = self._dry(other, st)
                self.assertFalse(any("offdec.py" in c for c in cmds), (other, st))
            out, c2 = self._dry(other, "offdec")
            self.assertEqual(c2, [], other)
            self.assertIn("nothing to do", out)

    def test_export_and_install_carry_the_decoders_for_this_config_only(self):
        T = _module(self.pkg / "train.py", "pkg_train_offdec_test")
        I = _module(self.pkg / "infer.py", "pkg_infer_offdec_test")
        dec, base = self.cfg(), self.cfg("final_A22li")
        tag = dec["d3d"]["dec"]["tag"]
        self.assertEqual(I.dec_tags(dec), {tag})
        self.assertEqual(I.dec_tags(base), set())
        self.assertEqual(I.recipe_tags(dec), I.recipe_tags(base))
        row = dict(group="offdec", tag=tag, path=f"offdec/x/dec_s40.pt", dest="fec_work/offdec/x/dec_s40.pt")
        self.assertTrue(T.export_required(row, I.recipe_tags(dec), set(), I.dec_tags(dec)))
        self.assertFalse(T.export_required(row, I.recipe_tags(base), set(), I.dec_tags(base)))
        self.assertFalse(T.export_required(row, I.recipe_tags(base), set()))     # the old 3-argument call
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "w" / "d3d" / "m").mkdir(parents=True)
            (d / "w" / "d3d" / "m" / "model.pt").write_bytes(b"member")
            h = hashlib.sha256(b"member").hexdigest()
            wman = {"files": [dict(group="d3d", tag="m", path="d3d/m/model.pt", dest="fec_work/models/m/model.pt",
                                   bytes=6, sha256=h),
                              dict(row, bytes=7, sha256=hashlib.sha256(b"decoder").hexdigest())]}
            self.assertEqual(I.install_weights(d / "work", d / "w", wman, {"m"}), 1)        # decoder row skipped
            with self.assertRaises(SystemExit):                                              # needed and missing
                I.install_weights(d / "work2", d / "w", wman, {"m", tag})
            (d / "w" / "offdec" / "x").mkdir(parents=True)
            (d / "w" / "offdec" / "x" / "dec_s40.pt").write_bytes(b"decoder")
            self.assertEqual(I.install_weights(d / "work3", d / "w", wman, {"m", tag}), 2)
            self.assertTrue((d / "work3" / "fec_work" / "offdec" / "x" / "dec_s40.pt").is_file())

    def test_inference_download_list_by_config(self):
        """Review rev_decpkg D1: reproduce.sh inference mode downloads infer.weight_paths(manifest, config): the decoder
        rows only for a config with d3d.dec, every other row unchanged for every config."""
        I = _module(self.pkg / "infer.py", "pkg_infer_offdec_test3")
        tag = self.cfg()["d3d"]["dec"]["tag"]
        rows = [dict(group="d3d", tag="m", path="d3d/m/model.pt"), dict(group="d3d", tag="x", path="d3d/x/model.pt"),
                dict(group="mast", tag="n7h_f0.55", path="mast/n7h_f0.55/model.pt"),
                dict(group="mast_cache", tag="passive_model", path="cache/p.npz")]
        rows += [dict(group="offdec", tag=tag, path=f"offdec/d3d_ip_C2_w3k22liA/dec_s{s}.pt") for s in (40, 41, 42)]
        wman = {"files": rows}
        dec_paths = {r["path"] for r in rows if r["group"] == "offdec"}
        other = sorted(r["path"] for r in rows if r["group"] != "offdec")
        for name in ("final_A22li", "final_A22li_r2", "final_B"):
            if (self.pkg / "configs" / f"{name}.json").exists():
                self.assertEqual(I.weight_paths(wman, self.cfg(name)), other, name)
        self.assertEqual(I.weight_paths(wman, self.cfg()), sorted(other + sorted(dec_paths)))
        f = REPO / "weights_manifest.json"                      # the real manifest (inside the package)
        if (REPO / "configs" / f"{CFG}.json").exists() and f.exists():
            w = json.loads(f.read_text())
            allp = sorted({r["path"] for r in w["files"]})
            dp = {r["path"] for r in w["files"] if r.get("group") == "offdec"}
            self.assertEqual(len(dp), 3)
            self.assertEqual(I.weight_paths(w, self.cfg("final_A22li")), sorted(set(allp) - dp))
            self.assertEqual(I.weight_paths(w, self.cfg()), allp)
        sh = self.pkg / "reproduce.sh"
        if not sh.exists():
            sh = REPO / "fec" / "final_pkg_tmpl" / "reproduce.sh"
        txt = sh.read_text(encoding="utf-8").replace(chr(13), "")
        nl = chr(10)
        inference = txt.split('if [ "$MODE" = full ]; then', 1)[1].split(nl + "else" + nl, 1)[1]
        inference = inference.split(nl + "fi" + nl, 1)[0]
        self.assertIn('python - "$WEIGHTS" "$CFG" <<', inference)
        self.assertIn("paths = weight_paths(w, json.load(open(f\"configs/{sys.argv[2]}.json\")))", inference)
        self.assertNotIn('paths = sorted({r["path"] for r in w["files"]})', inference)

    def test_peak_disk_per_config(self):
        T = _module(self.pkg / "train.py", "pkg_train_offdec_test2")
        self.assertEqual(T.peak_disk_gb("final_A22li"), T.PEAK_DISK_GB)
        self.assertEqual(T.peak_disk_gb("final_B"), T.PEAK_DISK_GB)
        self.assertEqual(T.peak_disk_gb(CFG), T.PEAK_DISK_GB + 1)

    def test_weights_manifest_rows(self):
        f = REPO / "weights_manifest.json"
        if not (REPO / "configs" / f"{CFG}.json").exists():
            self.skipTest("weights_manifest.json with the decoder rows is written by the package build")
        rows = [r for r in json.loads(f.read_text())["files"] if r.get("group") == "offdec"]
        q = self.cfg()["d3d"]["dec"]
        self.assertEqual([r["path"] for r in rows], [f"{q['dir']}/dec_s{s}.pt" for s in q["seeds"]])
        for r in rows:
            self.assertEqual(r["tag"], q["tag"])
            self.assertEqual(r["dest"], "fec_work/" + r["path"])
            self.assertRegex(r["sha256"] or "", r"^[0-9a-f]{64}$")


def _args(**kw):
    base = dict(split="public_test", subset_shots=0, root=None, dry_run=True, stage="all", d3d_work=None,
                owner="test", gpu_priority="normal", no_runlog=True, dec_dir=None, d3d_dir=None, q95h_src=None,
                force=False, workers=2, allow_unreviewed=False)
    base.update(kw)
    return SimpleNamespace(**base)


class TestR2Preset(_Pkg):
    """final_A22li_r2 (root scope addition 2026-10-14: the r = 2 knob adopted on Dev, EVIDENCE C66): final_A22li with
    the C1 passive resistance scale 2 for the amortized maps; the r05 pattern (agent fpkgm, B37)."""
    def test_recipe(self):
        r2, a22, r05 = self.cfg("final_A22li_r2"), self.cfg("final_A22li"), self.cfg("final_A22li_r05")
        self.assertEqual(r2["d3d"], a22["d3d"])
        m2, m05, m1 = (x["mast"]["composite"]["maps"] for x in (r2, r05, a22))
        self.assertEqual((m2["kind"], m2["rscale"]), ("amort", 2))
        self.assertNotIn("rscale", m1)
        self.assertEqual({k: v for k, v in m2.items() if k != "rscale"}, {k: v for k, v in m05.items() if k != "rscale"})
        for k in ("scalars", "betan_fill"):
            self.assertEqual(r2["mast"]["composite"][k], a22["mast"]["composite"][k])
        self.assertIn("v7h055r2tsm30", r2["name"])
        t2 = json.loads((self.pkg / "configs" / "train_d3d_final_A22li_r2.json").read_text())
        t1 = json.loads((self.pkg / "configs" / "train_d3d_final_A22li.json").read_text())
        self.assertEqual(t2["groups"], t1["groups"])

    def test_mast_amort_commands(self):
        import fec.run_final as RF
        for name in ("final_A22li_r2", CFG):
            with tempfile.TemporaryDirectory() as d:
                rec = json.loads((self.pkg / "configs" / f"{name}.json").read_text())
                c = RF.Ctx(_args(root=d, amort_net_dir=None, mast_scalars_dir=None, tx1s_model_dir=None, mast_dir=None,
                                 blim="recipe", psi_dtype=None), rec)
                RF.st_mast_amort(c)
                cmds = [x.split(" -- ")[-1] for x in c.cmds]
                D = RF.composite_dirs(c)
            self.assertEqual(len(cmds), 3, name)
            self.assertIn("passive_f", cmds[0])
            self.assertTrue(cmds[0].endswith("--rscales 2"), cmds[0])
            self.assertIn(" amort ", cmds[1])
            self.assertTrue(cmds[1].endswith("--rscale 2"), cmds[1])
            self.assertIn(" tsm ", cmds[2])
            self.assertEqual(D["amort"].name, "amort_n7h_f0.55_r2")
            self.assertEqual(D["maps"].name, "amort_n7h_f0.55_r2_tsm30")


class TestRunFinalOrder(_Pkg):
    def test_stage_order(self):
        import fec.run_final as RF
        rec, _ = RF.load_recipe(str(self.pkg / "configs" / f"{CFG}.json"), "C:/nonexistent_root_for_test")
        st = RF.all_stages(rec)
        i = st.index("d3d_q95h")
        self.assertEqual(st[i:i + 3], ["d3d_q95h", "d3d_dec", "d3d_lirep4"])
        self.assertNotIn("d3d_dec", RF.all_stages(json.loads((self.pkg / "configs" / "final_A22li.json").read_text())))
        bad = json.loads(json.dumps(rec))
        bad["d3d"]["dec"]["seeds"] = "40"
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "r.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                RF.load_recipe(str(Path(d) / "r.json"), d)

    def _manifest(self, c, name, info, status="ok"):
        c.L.manifest_dir.mkdir(parents=True, exist_ok=True)
        (c.L.manifest_dir / f"{name}.json").write_text(json.dumps(dict(status=status, info=info)))

    def test_lirep4_reads_the_decoded_preds(self):
        import fec.run_final as RF
        rec = json.loads((self.pkg / "configs" / f"{CFG}.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            c = RF.Ctx(_args(root=d), rec)
            src, out = RF.dec_dirs(c)
            self.assertEqual(src.name, "d3d_ip_C2_w3k22liA_q95h")
            self.assertEqual(out.name, "d3d_ip_C2_w3k22liA_q95h_dec")
            self.assertEqual(RF.lirep4_dirs(c)[0], out)                 # dry run: the dec dir
            c = RF.Ctx(_args(root=d, dry_run=False), rec)
            self._manifest(c, "d3d_q95h", dict(fallback=None))
            with self.assertRaises(RF.StageError):                        # dec not run yet
                RF.lirep4_dirs(c)
            self._manifest(c, "d3d_dec", dict(fallback=None))
            s4, o4 = RF.lirep4_dirs(c)
            self.assertEqual((s4.name, o4.name), ("d3d_ip_C2_w3k22liA_q95h_dec", "d3d_ip_C2_w3k22liA_q95h_dec_lirep4"))
            self._manifest(c, "d3d_dec", dict(fallback="src", error="test"))
            s4, o4 = RF.lirep4_dirs(c)
            self.assertEqual((s4.name, o4.name), ("d3d_ip_C2_w3k22liA_q95h", "d3d_ip_C2_w3k22liA_q95h_lirep4"))
            self.assertEqual(RF.dec_weights_dir(c), c.L.work / "offdec" / "d3d_ip_C2_w3k22liA")
            self.assertEqual(RF.dec_weights_dir(RF.Ctx(_args(root=d, dec_dir="X:/decs"), rec)), Path("X:/decs"))
        base = json.loads((self.pkg / "configs" / "final_A22li.json").read_text())
        with tempfile.TemporaryDirectory() as d:                          # no d3d.dec: the old rule, unchanged
            c = RF.Ctx(_args(root=d, dry_run=False), base)
            self._manifest(c, "d3d_q95h", dict(fallback=None))
            s4, o4 = RF.lirep4_dirs(c)
            self.assertEqual((s4.name, o4.name), ("d3d_ip_C2_w3k22liA_q95h", "d3d_ip_C2_w3k22liA_q95h_lirep4"))
            with self.assertRaises(RF.StageError):
                RF.dec_dirs(c)

    def test_done_tracks_decoders_and_code(self):
        """Review rev_decpkg D2: d3d_dec is done only for the same decoder files and fec/offdec.py."""
        import fec.run_final as RF
        rec = json.loads((self.pkg / "configs" / f"{CFG}.json").read_text())
        with tempfile.TemporaryDirectory() as d:
            dd = Path(d) / "decs"
            dd.mkdir()
            for s in (40, 41, 42):
                (dd / f"dec_s{s}.pt").write_bytes(f"decoder {s}".encode())
            c = RF.Ctx(_args(root=d, dry_run=False, dec_dir=str(dd)), rec)
            self._manifest(c, "d3d_q95h", dict(fallback=None))
            src, _ = RF.dec_dirs(c)
            info = dict(fallback="src", src=str(src), decoders=RF.dec_decoder_shas(c),
                        offdec_py_sha256=RF.sha256(RF.FEC / "offdec.py"))
            self.assertTrue(all(info["decoders"].values()))
            self._manifest(c, "d3d_dec", info)
            self.assertTrue(RF.done_d3d_dec(c))
            self._manifest(c, "d3d_dec", dict(info, offdec_py_sha256="0" * 64))      # other code
            self.assertFalse(RF.done_d3d_dec(c))
            self._manifest(c, "d3d_dec", info)
            (dd / "dec_s41.pt").write_bytes(b"retrained")                           # other decoder
            self.assertFalse(RF.done_d3d_dec(c))
            (dd / "dec_s41.pt").unlink()                                            # missing decoder
            self.assertIsNone(RF.dec_decoder_shas(c)["dec_s41.pt"])
            self.assertFalse(RF.done_d3d_dec(c))

    def test_dec_checks(self):
        import fec.run_final as RF
        q = self.cfg()["d3d"]["dec"]
        meta = dict(corr_rms_median=7.86e-6, corr_rms_p99=2.35e-5, nan=False, inspan_rms=1e-9, decoders=[1, 2, 3])
        self.assertTrue(all(RF.dec_checks(q, meta, "a", "a", 10, 10).values()))
        for k, v in (("nan", True), ("inspan_rms", 2e-8), ("corr_rms_median", 3e-5), ("corr_rms_p99", 1e-6)):
            self.assertFalse(all(RF.dec_checks(q, dict(meta, **{k: v}), "a", "a", 10, 10).values()), k)
        self.assertFalse(all(RF.dec_checks(q, meta, "a", "b", 10, 10).values()))        # scalars changed
        self.assertFalse(all(RF.dec_checks(q, meta, "a", "a", 9, 10).values()))


# ============================================================================ fec/offdec.py, synthetic, CPU
def _basis(seed=0):
    rng = np.random.default_rng(seed)
    V = np.linalg.qr(rng.standard_normal((4225, 128)))[0].astype(np.float32)
    m = (0.1 * rng.standard_normal(4225)).astype(np.float32)
    return m, V


def _save_member(d, tag, m, V):
    import torch
    (Path(d) / tag).mkdir(parents=True, exist_ok=True)
    torch.save({"mean": torch.tensor(m), "V": torch.tensor(V), "args": {"alldata": True}}, Path(d) / tag / "model.pt")


class TestOffdecUnits(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch absent")
        from fec import offdec
        self.O = offdec

    def test_basis_shared_by_all_members_else_refused(self):
        m, V = _basis()
        with tempfile.TemporaryDirectory() as d:
            _save_member(d, "a", m, V)
            _save_member(d, "b", m, V)
            V2 = V.copy(); V2[3, 5] += np.float32(5e-7)                # inside 1e-6
            _save_member(d, "c", m, V2)
            m64, V64, info = self.O.member_basis(d, ["a", "b", "c"], log=lambda s: None)
            self.assertTrue(np.array_equal(m64, m.astype(np.float64)) and np.array_equal(V64, V.astype(np.float64)))
            self.assertEqual(info["n_bit_identical"], 2)
            self.assertEqual(info["basis_sha256"], self.O.basis_sha256(m, V))
            V3 = V.copy(); V3[7, 1] += np.float32(5e-6)                # a fallback member 5e-6 off: refused
            _save_member(d, "fb", m, V3)
            with self.assertRaises(ValueError):
                self.O.member_basis(d, ["a", "b", "c", "fb"], log=lambda s: None)
            m4 = m.copy(); m4[0] += np.float32(1e-5)
            _save_member(d, "fm", m4, V)
            with self.assertRaises(ValueError):
                self.O.member_basis(d, ["a", "fm"], log=lambda s: None)
            with self.assertRaises(ValueError):
                self.O.member_basis(d, ["a", "missing"], log=lambda s: None)

    def _decoder_file(self, path, Qt, seed, mu, sd, rs=1.3e-5):
        import torch
        torch.manual_seed(seed)
        net = self.O.make_decoder(Qt)
        torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "rs": rs, "seed": seed, "epochs": 0,
                    "basis_sha256": None}, path)

    def test_apply_offspan_no_nan_scalars_copied(self):
        import torch
        m, V = _basis(1)
        M0, Q = m.astype(np.float64), V.astype(np.float64)
        rng = np.random.default_rng(2)
        n = 300
        psi = (m + (rng.standard_normal((n, 128)) * 0.05).astype(np.float32) @ V.T +
               1e-5 * rng.standard_normal((n, 4225)).astype(np.float32)).astype(np.float32).reshape(n, 65, 65)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src, out = d / "src", d / "src_dec"
            src.mkdir()
            np.save(src / "cfg_psi.npy", psi)
            np.save(src / "cfg_scal.npy", rng.standard_normal((n, 5)).astype(np.float32))
            (src / "cfg_meta.json").write_text("{}")
            Qt = torch.tensor(Q, dtype=torch.float32)
            c = ((psi.reshape(n, -1).astype(np.float64) - M0) @ Q).astype(np.float32)
            decs = []
            for s in (40, 41, 42):
                p = d / f"dec_s{s}.pt"
                self._decoder_file(p, Qt, s, c.mean(0), c.std(0))
                decs.append(p)
            bsha = self.O.basis_sha256(m, V)
            meta = self.O.apply_decoders(src, out, "cfg", decs, M0, Q, "cpu", chunk=128, expect_basis=bsha,
                                         basis_sha=bsha, log=lambda s: None)
            y = np.load(out / "cfg_psi.npy")
            corr = y.reshape(n, -1).astype(np.float64) - psi.reshape(n, -1)
            self.assertFalse(meta["nan"])
            self.assertTrue(np.isfinite(y).all())
            self.assertLess(meta["inspan_rms"], 1e-8)
            self.assertLess(meta["inspan_max_abs"], 1e-8)
            self.assertGreater(meta["corr_rms_median"], 0)
            # the stored psi's in-span change is float32 rounding of psi only (the correction itself is off-span)
            self.assertLess(float(np.sqrt(((corr @ Q) ** 2).mean())), 1e-7)
            for f in ("cfg_scal.npy", "cfg_meta.json"):
                self.assertEqual((src / f).read_bytes(), (out / f).read_bytes(), f)
            self.assertEqual(meta["copied_unchanged"], ["cfg_meta.json", "cfg_scal.npy"])
            with self.assertRaises(SystemExit):                              # never overwrites
                self.O.apply_decoders(src, out, "cfg", decs, M0, Q, "cpu", log=lambda s: None)
            with self.assertRaises(ValueError):                              # decoders of another basis
                self.O.apply_decoders(src, d / "o2", "cfg", decs, M0, Q, "cpu", expect_basis="0" * 64,
                                      basis_sha=bsha, log=lambda s: None)

    def _cache(self, d, n=4400):
        """A tiny synthetic offdec cache (member_basis, envelope, tr_c / tr_r / tr_rms / tr_wp)."""
        m, V = _basis(3)
        rng = np.random.default_rng(4)
        np.savez(d / "member_basis.npz", m=m.astype(np.float64), V=V.astype(np.float64))
        (d / "member_basis.json").write_text(json.dumps({"basis_sha256": self.O.basis_sha256(m, V)}))
        th = np.linspace(-np.pi, np.pi, 720, endpoint=False)
        np.save(d / "d3d_env_m4.npy", np.c_[1.7 + 0.8 * np.cos(th), 1.3 * np.sin(th)])
        np.save(d / "tr_c.npy", rng.standard_normal((n, 128)).astype(np.float32))
        r = 1e-5 * rng.standard_normal((n, 4225))
        np.save(d / "tr_r.npy", (r * 1e4).astype(np.float16))
        rms = np.sqrt((r ** 2).mean(1)); rms[5] = 1e-3                     # one artefact frame (dropped)
        np.save(d / "tr_rms.npy", rms)
        wp = np.tile(np.array([1.0, 0.0, 1.5, 1.0, 1.5, -1.0], np.float32), (n, 1)); wp[7] = np.nan
        np.save(d / "tr_wp.npy", wp)

    def test_train_deterministic_given_the_seed_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self._cache(d)
            log = lambda s: None  # noqa: E731
            a = self.O.train_decoder(d, 40, d / "a.pt", epochs=1, device="cpu", log=log)
            b = self.O.train_decoder(d, 40, d / "b.pt", epochs=1, device="cpu", log=log)
            c = self.O.train_decoder(d, 41, d / "c.pt", epochs=1, device="cpu", log=log)
            self.assertEqual(a["n_kept"], 4399)
            self.assertEqual(a["basis_sha256"], json.loads((d / "member_basis.json").read_text())["basis_sha256"])
            same = all(np.array_equal(a["state"][k].numpy(), b["state"][k].numpy()) for k in a["state"])
            self.assertTrue(same)
            self.assertTrue(np.array_equal(a["mu"], b["mu"]) and a["rs"] == b["rs"])
            self.assertFalse(all(np.array_equal(a["state"][k].numpy(), c["state"][k].numpy()) for k in a["state"]))
            import torch
            k = torch.load(d / "a.pt", weights_only=False)
            self.assertEqual(sorted(k)[:7], sorted(["basis", "basis_sha256", "epochs", "final_loss", "max_frames",
                                                    "mu", "n_kept"]))
            for key in ("state", "mu", "sd", "rs", "seed", "epochs"):                # what the apply step reads
                self.assertIn(key, k)


if __name__ == "__main__":
    unittest.main()
