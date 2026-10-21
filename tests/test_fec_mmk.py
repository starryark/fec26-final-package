"""final_MTfull_mmk = final_MTfull + the member-mean kappa finishing stage d3d_mmk (fec/mmk.py, frozen operator
fec/mmk_op.py; agent mmks2 2026-10-19).

Shipped in the Final package (tests/) and run in the research repo.  CPU only, synthetic Solov'ev maps with a synthetic
envelope (no challenge data):
  * the operator file is the frozen one (SHA-256) and its constants are the recipe's; only the kappa step is called;
  * the kappa step: identity when the target equals kappa, a half-way move written as float32 whose re-extracted kappa
    is the recorded realised value (and survives an .npy round trip), unreachable / NaN targets leave the frame
    bit-identical;
  * `fec/mmk.py apply`: targets = the members' equal NaN-ignoring mean; the no-Thomson fallback rows are never edited;
    skipped frames are bit-identical; the scalar file is copied byte for byte; the output does not depend on the worker
    count; frames are edited independently (a prefix or a reversed order gives the same frames);
  * inputs only: the code names no scorer mask, no EFIT / label field, and the stage reads only the files it is given;
  * the d3d_b hook (run_final.predict_b_chunked): the ensemble files are byte-identical with and without it, the kappa
    file holds every member's own kappa (also on fallback rows), the fallback rows are recorded;
  * run_final wiring: final_MTfull_mmk = final_MTfull + d3d.mmk; q95h -> dec -> mmk -> lirep4; final_MTfull and
    final_MTli keep their stages and commands.
Amendment 59e hardening (agent mmk2pkg, 2026-10-20):
  * worker pools: a dead worker, a worker error and a stall (no task finished within the timeout) end the mmk part;
    the hook in d3d_b then writes no kappa file and leaves the ensemble byte-identical (also for import / setup /
    initializer / close / save errors, and stale files of an earlier run are removed); `apply` raises and writes no
    output map, so d3d_mmk records fallback=src;
  * stale-kappa guard: d3d_mmk falls back unless the kappa meta's run id, input fingerprint and ensemble SHA-256 are
    the current run's;
  * a frame with fewer than min_members finite member kappa is not edited (status 11, counted);
  * an mmk fallback is a submission-level fallback (d3d_submit_dir marker "mmk_src");
  * operator registry: every registered operator is fec/mmk_op.py with at most ALPHA changed; make-operator writes one.
"""
import ast
import builtins
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fec import mmk as MMK  # noqa: E402
from fec import mmk_op as OP  # noqa: E402
from fec.paths import SCORING  # noqa: E402

R = np.linspace(0.84, 2.54, 65)
Z = np.linspace(-1.6, 1.6, 65)
RR, ZZ = np.meshgrid(R, Z)
MASK = np.zeros((65, 65), bool)
MASK[3:-3, 3:-3] = True


def solovev(R0=1.70, kap=1.7, A=1.5, Rc=1.55, Zc=-1.45, w=0.18):
    """Solov'ev flux (axis minimum, DIII-D sign) plus a Gaussian 'divertor coil' well that makes a lower X-point."""
    p = RR ** 2 * ZZ ** 2 / kap ** 2 + (RR ** 2 - R0 ** 2) ** 2 / 4.0
    return (p - A * np.exp(-((RR - Rc) ** 2 + (ZZ - Zc) ** 2) / (2 * w * w))).astype(np.float32)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _func():
    MMK._init_worker(_Env.path, SCORING)            # the worker state, in this process
    return MMK._W["func"]


class _Env:
    """A synthetic TRAIN-envelope file (the format of env_train.npz) shared by the tests."""
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "env_train.npz"
    np.savez(path, grid_R=R, grid_Z=Z, env=MASK)


def _kappa(func, p32):
    return MMK.frame_kappa(func, p32)


class TestFrozenOperator(unittest.TestCase):
    def test_operator_file_is_the_frozen_one(self):
        self.assertEqual(_sha(REPO / "fec" / "mmk_op.py"), MMK.OP_SHA256)
        self.assertEqual(MMK.op_constants(), MMK.FROZEN)
        self.assertEqual(MMK.check_frozen(), [])
        self.assertEqual((OP.ALPHA, OP.SIGMA_Z, tuple(OP.PSIN_TAPER), tuple(OP.C_BRACKET)), (0.5, 0.25, (1.0, 1.4),
                                                                                              (0.9, 1.1)))

    def test_recipe_block_is_checked(self):
        import fec.run_final as RF
        q = RF.RECIPES["final_MTfull_mmk"]["d3d"]["mmk"]
        self.assertEqual(MMK.check_frozen(q), [])
        self.assertEqual(q["env_train_sha256"], MMK.ENV_TRAIN_SHA256)
        for k, v in (("alpha", 0.6), ("c_bracket", [0.8, 1.2]), ("operator_sha256", "0" * 64), ("step", "delta"),
                     ("env_train_sha256", "1" * 64)):
            self.assertTrue(MMK.check_frozen(dict(q, **{k: v})), k)

    def test_only_the_kappa_step_is_called(self):
        src = (REPO / "fec" / "mmk.py").read_text(encoding="utf-8")
        calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id in ("OP", "op")]
        names = sorted({n.func.attr for n in calls})
        self.assertEqual(names, ["EnvFunctional", "_step", "load_env"])
        steps = [n for n in calls if n.func.attr == "_step"]
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].args[3].value, "k")


class TestKappaStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f = _func()
        cls.p32 = solovev()
        cls.k0 = _kappa(cls.f, cls.p32)

    def test_synthetic_map_is_sane(self):
        self.assertTrue(1.5 < self.k0 < 2.0)

    def test_identity_when_target_equals_kappa(self):
        a, fl = MMK.edit_frame(self.f, self.p32, self.k0)
        self.assertEqual(fl["k_status"], 1)
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(a.tobytes(), self.p32.tobytes())

    def test_half_way_move_and_float32_round_trip(self):
        T = self.k0 + 0.02
        a, fl = MMK.edit_frame(self.f, self.p32, T)
        self.assertEqual(fl["k_status"], 0)
        self.assertEqual(a.dtype, np.float32)
        self.assertNotEqual(a.tobytes(), self.p32.tobytes())
        self.assertEqual(_kappa(self.f, a), fl["k_real"])                 # the written float32 map is what was checked
        self.assertLess(abs(fl["k_real"] - (self.k0 + 0.01)), 1e-4)
        self.assertEqual(fl["k_target"], self.k0 + MMK.FROZEN["alpha"] * (T - self.k0))
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d) / "x.npy", a[None])
            b = np.load(Path(d) / "x.npy")[0]
        self.assertEqual(b.tobytes(), a.tobytes())
        self.assertEqual(_kappa(self.f, b), fl["k_real"])
        a2, fl2 = MMK.edit_frame(self.f, self.p32, T)                       # deterministic
        self.assertEqual(a2.tobytes(), a.tobytes())

    def test_unreachable_and_nan_targets_leave_the_frame(self):
        for T, st in ((self.k0 + 2.0, 4), (np.nan, 9)):
            a, fl = MMK.edit_frame(self.f, self.p32, T)
            self.assertEqual(fl["k_status"], st)
            self.assertEqual(a.tobytes(), self.p32.tobytes())


def _write_src(d, P, scal=None):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "cfg_psi.npy", np.asarray(P, np.float32))
    np.save(d / "cfg_scal.npy", np.asarray(scal if scal is not None else
                                           np.arange(len(P) * 5, dtype=np.float32).reshape(-1, 5)))
    (d / "cfg_dec_meta.json").write_text('{"x": 1}')
    return d


class TestApply(unittest.TestCase):
    """fec/mmk.py apply on 6 synthetic frames with 3 members' kappa."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        f = _func()
        kaps = [1.70, 1.72, 1.68, 1.75, 1.66, 1.71]
        cls.P = np.stack([solovev(kap=k) for k in kaps])
        mem = [[solovev(kap=k + dk) for dk in (-0.04, 0.0, 0.07)] for k in kaps]
        cls.K = np.array([[_kappa(f, m) for m in row] for row in mem])
        cls.K[3] = np.nan                                           # every member failed on frame 3 -> T NaN
        cls.K[5, 1] = np.nan                                        # one member failed on frame 5 -> mean of 2
        np.save(cls.d / "kappa.npy", cls.K)
        np.save(cls.d / "skip.npy", np.array([4], np.int64))       # the fallback wrote frame 4
        cls.src = _write_src(cls.d / "src", cls.P)
        cls.m1 = MMK.apply(cls.src, cls.d / "o1", "cfg", cls.d / "kappa.npy", _Env.path, nproc=1, chunk=2,
                           skip_rows=cls.d / "skip.npy", log=lambda m: None)
        cls.m2 = MMK.apply(cls.src, cls.d / "o2", "cfg", cls.d / "kappa.npy", _Env.path, nproc=2, chunk=4,
                           skip_rows=cls.d / "skip.npy", log=lambda m: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_targets_are_the_members_mean(self):
        fl = np.load(self.d / "o1" / "cfg_mmk_flags.npz")
        T = fl["T_kappa"]
        self.assertTrue(np.isnan(T[3]))
        self.assertEqual(T[5], np.nanmean(self.K[5]))
        self.assertEqual(T[0], np.mean(self.K[0]))
        self.assertTrue(np.array_equal(T, MMK.member_targets(self.K), equal_nan=True))

    def test_statuses_skips_and_bytes(self):
        P1 = np.load(self.d / "o1" / "cfg_psi.npy")
        st = np.load(self.d / "o1" / "cfg_mmk_flags.npz")["k_status"]
        self.assertEqual(st[3], 9)                                   # target NaN
        self.assertEqual(st[4], MMK.FALLBACK_FRAME)                  # fallback frame: never edited
        self.assertEqual(self.m1["fallback_frames_not_edited"], 1)
        self.assertEqual(self.m1["status_counts"]["fallback_frame"], 1)
        self.assertEqual(self.m1["skip_counts"]["target_nan"], 1)
        changed = [P1[i].tobytes() != self.P[i].tobytes() for i in range(6)]
        self.assertEqual(changed, list(st == 0))
        self.assertTrue(self.m1["edited_rows_equal_ok_rows"])
        self.assertGreaterEqual(int((st == 0).sum()), 3)
        self.assertEqual(P1.dtype, np.float32)
        self.assertEqual(_sha(self.d / "o1" / "cfg_scal.npy"), _sha(self.src / "cfg_scal.npy"))
        self.assertEqual(_sha(self.d / "o1" / "cfg_dec_meta.json"), _sha(self.src / "cfg_dec_meta.json"))
        self.assertTrue(self.m1["finite"])

    def test_worker_count_and_chunking_do_not_change_the_output(self):
        self.assertEqual(_sha(self.d / "o1" / "cfg_psi.npy"), _sha(self.d / "o2" / "cfg_psi.npy"))
        f1, f2 = (np.load(self.d / o / "cfg_mmk_flags.npz") for o in ("o1", "o2"))
        for k in MMK.FLAG_KEYS:
            self.assertTrue(np.array_equal(f1[k], f2[k], equal_nan=True), k)

    def test_frames_are_edited_independently(self):
        """No other frame (later or earlier) enters a frame's edit: a prefix and a reversed order give the same rows."""
        P1 = np.load(self.d / "o1" / "cfg_psi.npy")
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            np.save(t / "k3.npy", self.K[:3])
            MMK.apply(_write_src(t / "s3", self.P[:3]), t / "o3", "cfg", t / "k3.npy", _Env.path, nproc=1,
                      log=lambda m: None)
            self.assertEqual(np.load(t / "o3" / "cfg_psi.npy").tobytes(), P1[:3].tobytes())
            np.save(t / "kr.npy", self.K[::-1].copy())
            np.save(t / "sr.npy", np.array([1], np.int64))          # frame 4 is row 1 when reversed
            MMK.apply(_write_src(t / "sr", self.P[::-1].copy()), t / "or", "cfg", t / "kr.npy", _Env.path, nproc=1,
                      skip_rows=t / "sr.npy", log=lambda m: None)
            self.assertEqual(np.load(t / "or" / "cfg_psi.npy")[::-1].tobytes(), P1.tobytes())

    def test_never_overwrites(self):
        with self.assertRaises(SystemExit):
            MMK.apply(self.src, self.d / "o1", "cfg", self.d / "kappa.npy", _Env.path, nproc=1, log=lambda m: None)


class TestInputsOnly(unittest.TestCase):
    FORBIDDEN = ("d3d_envelope", "mask_coarse", "masks", "efit", "psirz", "diii_d_test", "private_test_label",
                 "y_psi", "labels")

    def _code_tokens(self, path):
        t = ast.parse(Path(path).read_text(encoding="utf-8"))
        doc = set()
        for n in ast.walk(t):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and \
                    isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                doc.add(id(n.body[0].value))
        out = []
        for n in ast.walk(t):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc:
                out.append(n.value)
            elif isinstance(n, ast.Name):
                out.append(n.id)
            elif isinstance(n, ast.Attribute):
                out.append(n.attr)
        return out

    def test_code_names_no_mask_or_label(self):
        for f in ("mmk.py", "mmk_op.py"):
            toks = " ".join(self._code_tokens(REPO / "fec" / f)).lower()
            for bad in self.FORBIDDEN:
                self.assertNotIn(bad, toks, f)

    def test_reads_only_the_given_files(self):
        """The parent reads src / kappa / skip rows / envelope; a worker reads the envelope (and code) only."""
        seen = []
        real_open, real_load = builtins.open, np.load

        def rec_open(file, *a, **k):
            if isinstance(file, (str, Path)):                      # (not an int file descriptor)
                seen.append(str(file))
            return real_open(file, *a, **k)

        def rec_load(file, *a, **k):
            seen.append(str(file))
            return real_load(file, *a, **k)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            P = np.stack([solovev(kap=1.7), solovev(kap=1.74)])
            src = _write_src(t / "src", P)
            np.save(t / "k.npy", np.array([[1.72, 1.74], [1.70, 1.71]]))
            np.save(t / "s.npy", np.array([], np.int64))
            with mock.patch.object(builtins, "open", rec_open), mock.patch.object(np, "load", rec_load):
                MMK._init_worker(_Env.path, SCORING)
                key, out, fl = MMK._apply_task((0, P, np.array([1.73, 1.70]), np.zeros(2, bool)))
                MMK.apply(src, t / "out", "cfg", t / "k.npy", _Env.path, nproc=1, skip_rows=t / "s.npy",
                          log=lambda m: None)
            data = [s for s in seen if not s.endswith(".py") and "__pycache__" not in s and "numba" not in s.lower()]
            allowed = (str(t), str(_Env.path))
            self.assertTrue(data)
            for s in data:
                self.assertTrue(s.startswith(allowed), s)


class TestMemberKappaHook(unittest.TestCase):
    """run_final.predict_b_chunked with the predictor mocked: 3 members, 1 no-Thomson fallback model, 6 frames."""
    KAP = {"m_a": 1.64, "m_b": 1.72, "m_c": 1.80, "fb": 1.50}

    @classmethod
    def _member(cls, ds, config, kind, tag, cache, rows=None):
        idx = np.arange(6)[rows if rows is not None else slice(None)]
        P = np.stack([solovev(kap=cls.KAP[tag] + 0.01 * i) for i in idx]).reshape(len(idx), -1)
        Q = np.tile(np.array([3.0, 2.0, 1.0, 1.7, 0.0], np.float32) * cls.KAP[tag], (len(idx), 1))
        return P.astype(np.float32), Q.astype(np.float32)

    def test_ensemble_unchanged_and_kappa_of_each_member(self):
        import fec.run_final as RF
        nots = np.zeros(6, bool)
        nots[2] = True
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            rec = t / "members.json"
            rec.write_text(json.dumps({"members": [{"kind": "b", "tag": x, "weight": 1.0} for x in ("m_a", "m_b", "m_c")],
                                       "fallback": [{"kind": "b", "tag": "fb", "weight": 1.0}]}))
            with mock.patch("fec.predict_d3d_b._member", self._member), \
                    mock.patch("fec.features.load_ds", lambda cfg: {"info": {"N": 6}}), \
                    mock.patch("fec.features_b.no_thomson_frames", lambda ds: nots):
                RF.predict_b_chunked("cfg", t / "plain", rec, chunk_rows=4096, log=lambda m: None)
                RF.predict_b_chunked("cfg", t / "mmk", rec, chunk_rows=4096, log=lambda m: None,
                                     mmk=dict(env=str(_Env.path), nproc=2))
            for f in ("cfg_psi.npy", "cfg_scal.npy"):
                self.assertEqual(_sha(t / "plain" / f), _sha(t / "mmk" / f), f)
            self.assertFalse((t / "plain" / "cfg_mmk_kappa.npy").exists())
            K = np.load(t / "mmk" / "cfg_mmk_kappa.npy")
            meta = json.loads((t / "mmk" / "cfg_mmk_kappa.json").read_text())
            self.assertEqual(meta["tags"], ["m_a", "m_b", "m_c"])
            self.assertTrue(meta["complete"])
            self.assertEqual(meta["sha256"], _sha(t / "mmk" / "cfg_mmk_kappa.npy"))
            self.assertEqual(np.load(t / "mmk" / "cfg_mmk_fallback_rows.npy").tolist(), [2])
            self.assertEqual(meta["fallback_rows"]["n"], 1)
            f = _func()
            want = np.array([[_kappa(f, self._member(None, None, "b", tag, None, rows=np.array([i]))[0][0])
                              for tag in ("m_a", "m_b", "m_c")] for i in range(6)])
            self.assertTrue(np.array_equal(K, want))                  # members' own maps, also on the fallback row 2
            m = json.loads((t / "mmk" / "cfg_meta.json").read_text())
            self.assertEqual(m["mmk_kappa"]["tags"], ["m_a", "m_b", "m_c"])
            self.assertNotIn("mmk_kappa", json.loads((t / "plain" / "cfg_meta.json").read_text()))


def _args(**kw):
    base = dict(split="private_test", subset_shots=0, root=None, dry_run=True, stage="all", d3d_work=None,
                owner="test", gpu_priority="normal", no_runlog=True, dec_dir=None, d3d_dir=None, q95h_src=None,
                force=False, workers=5, allow_unreviewed=False, no_model_hash=True)
    base.update(kw)
    return SimpleNamespace(**base)


class TestRunFinalWiring(unittest.TestCase):
    def test_recipe_is_mtfull_plus_mmk(self):
        import fec.run_final as RF
        a, b = RF.RECIPES["final_MTfull_mmk"], RF.RECIPES["final_MTfull"]
        a2 = json.loads(json.dumps(a))
        q = a2["d3d"].pop("mmk")
        a2.pop("name")
        b2 = json.loads(json.dumps(b))
        b2.pop("name")
        self.assertEqual(a2, b2)
        self.assertEqual(q, RF.MMK_KAPPA)
        self.assertIn("_dec_mmk_lirep4__", a["name"])
        self.assertEqual(a["d3d"]["b"]["name"], b["d3d"]["b"]["name"])
        self.assertEqual(a["mast"], b["mast"])
        self.assertNotIn("mmk", b["d3d"])
        self.assertNotIn("mmk", RF.RECIPES["final_MTli"]["d3d"])

    def test_stage_order(self):
        import fec.run_final as RF
        st = RF.all_stages(RF.RECIPES["final_MTfull_mmk"])
        i = st.index("d3d_q95h")
        self.assertEqual(st[i:i + 4], ["d3d_q95h", "d3d_dec", "d3d_mmk", "d3d_lirep4"])
        self.assertEqual([s for s in st if s != "d3d_mmk"], RF.all_stages(RF.RECIPES["final_MTfull"]))
        for name, r in RF.RECIPES.items():
            if not r["d3d"].get("mmk"):
                self.assertNotIn("d3d_mmk", RF.all_stages(r), name)
        bad = json.loads(json.dumps(RF.RECIPES["final_MTfull_mmk"]))
        bad["d3d"]["b"]["chunk_rows"] = 0
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "r.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                RF.load_recipe(str(Path(d) / "r.json"), d)

    def _cmds(self, name, stages, d):
        import fec.run_final as RF
        c = RF.Ctx(_args(root=d), json.loads(json.dumps(RF.RECIPES[name])))
        for s in stages:
            RF.STAGES[s][0](c)
        return [x.split(" -- ")[-1] for x in c.cmds if not x.startswith("link ")]

    def test_dry_run_commands(self):
        stages = ["d3d_b", "d3d_q95h", "d3d_dec", "d3d_lirep4"]
        with tempfile.TemporaryDirectory() as d:                     # dry runs write nothing: one root for both
            mt = self._cmds("final_MTfull", stages, d)
            mk = self._cmds("final_MTfull_mmk", stages[:3] + ["d3d_mmk"] + stages[3:], d)
        self.assertFalse(any("--mmk" in x or "mmk.py" in x or "_mmk" in x for x in mt))
        self.assertTrue(mk[0].endswith("--mmk-nproc 5 --mmk-run-id dry-run --mmk-stall-timeout 600.0"), mk[0])
        self.assertIn("--mmk-env ", mk[0])
        self.assertEqual(mk[0].split(" --mmk-env ")[0].replace(d, ""), mt[0].replace(d, ""))
        self.assertEqual([x.replace(d, "") for x in mk[1:3]], [x.replace(d, "") for x in mt[1:3]])
        self.assertIn("mmk.py apply", mk[3])
        for k in ("--skip-rows", "_mmk_fallback_rows.npy", "_mmk_kappa.npy", "env_train.npz", "--nproc 5",
                  "d3d_ip_C2_w3k22liAmtfull_q95h_dec_mmk", "--min-members 18", "--operator mmk_op",
                  "--stall-timeout 600.0"):
            self.assertIn(k, mk[3])
        self.assertIn("--src " + str(Path(d) / "preds" / "d3d_ip_C2_w3k22liAmtfull_q95h_dec_mmk").replace("\\", "/"),
                      mk[4].replace("\\", "/"))
        self.assertIn("d3d_ip_C2_w3k22liAmtfull_q95h_dec ", mt[3].replace("\\", "/") + " ")

    def test_lirep4_source_and_fallback(self):
        import fec.run_final as RF
        rec = json.loads(json.dumps(RF.RECIPES["final_MTfull_mmk"]))
        with tempfile.TemporaryDirectory() as d:
            c = RF.Ctx(_args(root=d, dry_run=False), rec)
            c.L.manifest_dir.mkdir(parents=True, exist_ok=True)

            def man(name, info):
                (c.L.manifest_dir / f"{name}.json").write_text(json.dumps(dict(status="ok", info=info)))
            man("d3d_q95h", dict(fallback=None))
            man("d3d_dec", dict(fallback=None))
            with self.assertRaises(RF.StageError):                 # mmk not run yet
                RF.lirep4_dirs(c)
            man("d3d_mmk", dict(fallback=None))
            s4, o4 = RF.lirep4_dirs(c)
            self.assertEqual((s4.name, o4.name), ("d3d_ip_C2_w3k22liAmtfull_q95h_dec_mmk",
                                                  "d3d_ip_C2_w3k22liAmtfull_q95h_dec_mmk_lirep4"))
            man("d3d_mmk", dict(fallback="src", error="test"))
            s4, o4 = RF.lirep4_dirs(c)
            self.assertEqual((s4.name, o4.name), ("d3d_ip_C2_w3k22liAmtfull_q95h_dec",
                                                  "d3d_ip_C2_w3k22liAmtfull_q95h_dec_lirep4"))
            self.assertEqual(RF.mmk_kappa_path(c).name, "diii_d_private_test_mmk_kappa.npy")
            self.assertEqual(RF.mmk_kappa_path(c).parent.name, "d3d_ip_C2_w3k22liAmtfull")
            self.assertEqual(len(RF.mmk_recipe_tags(c)), 22)
        with tempfile.TemporaryDirectory() as d:                      # final_MTfull: the old rule
            c = RF.Ctx(_args(root=d, dry_run=False), json.loads(json.dumps(RF.RECIPES["final_MTfull"])))
            c.L.manifest_dir.mkdir(parents=True, exist_ok=True)
            for n in ("d3d_q95h", "d3d_dec"):
                (c.L.manifest_dir / f"{n}.json").write_text(json.dumps(dict(status="ok", info=dict(fallback=None))))
            s4, o4 = RF.lirep4_dirs(c)
            self.assertEqual(s4.name, "d3d_ip_C2_w3k22liAmtfull_q95h_dec")
            with self.assertRaises(RF.StageError):
                RF.mmk_dirs(c)

    def _stage(self, d, env=None, nan_members=0, tamper=None):
        """A Ctx of final_MTfull_mmk with a consistent synthetic run in root d: q95h / dec manifests, the d3d_b manifest
        with its run id, the d3d_b ensemble files, the post-decoder source (3 frames), the members' kappa (22) with a
        meta that matches (tamper(meta) may break it) and fallback row 1.  env: the envelope file (its SHA-256 stands in
        for the frozen env_train's).  Returns (c, src, out, P, patches)."""
        import fec.run_final as RF
        env = Path(env or _Env.path)
        esha = _sha(env)
        rec = json.loads(json.dumps(RF.RECIPES["final_MTfull_mmk"]))
        rec["d3d"]["mmk"]["env_train_sha256"] = esha
        c = RF.Ctx(_args(root=d, dry_run=False, workers=1), rec)
        c.L.manifest_dir.mkdir(parents=True, exist_ok=True)
        for n in ("d3d_q95h", "d3d_dec"):
            (c.L.manifest_dir / f"{n}.json").write_text(json.dumps(dict(status="ok", info=dict(fallback=None))))
        (c.L.manifest_dir / "d3d_b.json").write_text(json.dumps(dict(status="ok", info=dict(mmk=dict(run_id="r1")))))
        src, out = RF.mmk_dirs(c)
        P = np.stack([solovev(kap=k) for k in (1.70, 1.66, 1.74)])
        cfg = c.L.d3d_cfg
        _write_src(src, P)
        for f in ("cfg_psi.npy", "cfg_scal.npy"):
            (src / f).rename(src / f.replace("cfg", cfg))
        f = _func()
        K = np.array([[_kappa(f, p) + 0.02 + 0.001 * m for m in range(22)] for p in P])
        if nan_members:
            K[2, :nan_members] = np.nan
        kap, fbr = RF.mmk_kappa_path(c), RF.mmk_fallback_rows_path(c)
        kap.parent.mkdir(parents=True, exist_ok=True)
        np.save(kap, K)
        np.save(fbr, np.array([1], np.int64))
        for k in ("psi", "scal"):                                   # d3d_b's ensemble files
            np.save(kap.parent / f"{cfg}_{k}.npy", np.load(src / f"{cfg}_{k}.npy"))
        n_frames = 3
        _, ish = RF.mmk_inputs(cfg, RF.resolve_path(rec["d3d"]["b"]["recipe"], RF.REPO), n_frames,
                               RF.mmk_recipe_tags(c), c.L.d3d_pred_work / "ds" / cfg / "info.json")
        meta = dict(tags=RF.mmk_recipe_tags(c), env_train_sha256=esha, sha256=_sha(kap), complete=True, run_id="r1",
                    inputs_sha256=ish, ensemble_psi_sha256=_sha(kap.parent / f"{cfg}_psi.npy"),
                    ensemble_scal_sha256=_sha(kap.parent / f"{cfg}_scal.npy"))
        if tamper:
            tamper(meta)
        MMK.write_json(MMK.kappa_meta_path(kap), meta)
        patches = (mock.patch.object(RF, "mmk_env", lambda c: (env, {})),
                   mock.patch.object(RF, "d3d_pred_N", lambda c: n_frames),
                   mock.patch.object(MMK, "ENV_TRAIN_SHA256", esha))
        return c, src, out, P, patches

    def _run_stage(self, *a, **kw):
        import fec.run_final as RF
        with tempfile.TemporaryDirectory() as d:
            c, src, out, P, patches = self._stage(d, *a, **kw)
            with patches[0], patches[1], patches[2]:
                info = RF.st_d3d_mmk(c)
                res = dict(info=info, P=P, cfg=c.L.d3d_cfg,
                           Q=np.load(out / f"{c.L.d3d_cfg}_psi.npy") if (out / f"{c.L.d3d_cfg}_psi.npy").exists() else None)
                if info.get("fallback") is None:
                    (c.L.manifest_dir / "d3d_mmk.json").write_text(json.dumps(dict(status="ok", info=info)))
                    res["done"] = RF.done_d3d_mmk(c)
                    np.save(RF.mmk_fallback_rows_path(c), np.array([2], np.int64))   # other fallback rows -> not done
                    res["done_after_change"] = RF.done_d3d_mmk(c)
                (c.L.manifest_dir / "d3d_lirep4.json").write_text(json.dumps(dict(status="ok", info=dict(fallback=None))))
                (c.L.manifest_dir / "d3d_mmk.json").write_text(json.dumps(dict(status="ok", info=info)))
                res["submit_marker"] = RF.d3d_submit_dir(c)[1]
            return res

    def test_stage_runs_the_cli_and_checks_it(self):
        """st_d3d_mmk end to end (the `fec/mmk.py apply` subprocess) on 3 synthetic frames with a synthetic envelope
        (its SHA-256 stands in for the frozen env_train's): edited frames, the fallback frame untouched, checks true."""
        r = self._run_stage()
        info, P, Q = r["info"], r["P"], r["Q"]
        self.assertIsNone(info["fallback"], info.get("error"))
        self.assertTrue(all(info["checks"].values()), info["checks"])
        self.assertEqual(info["status_counts"]["ok"], 2)
        self.assertEqual(info["status_counts"]["fallback_frame"], 1)
        self.assertEqual(info["fallback_frames_not_edited"], 1)
        self.assertEqual((info["min_members"], info["too_few_members_frames"]), (18, 0))
        self.assertEqual(Q[1].tobytes(), P[1].tobytes())
        self.assertNotEqual(Q[0].tobytes(), P[0].tobytes())
        self.assertTrue(r["done"])
        self.assertFalse(r["done_after_change"])
        self.assertIsNone(r["submit_marker"])

    def test_stage_leaves_frames_with_too_few_members(self):
        """59e: frame 2 has 5 NaN member kappa (17 of 22 finite < 18): status 11, unedited, counted."""
        r = self._run_stage(nan_members=5)
        info, P, Q = r["info"], r["P"], r["Q"]
        self.assertIsNone(info["fallback"], info.get("error"))
        self.assertEqual(info["too_few_members_frames"], 1)
        self.assertEqual(info["status_counts"]["too_few_members"], 1)
        self.assertEqual(info["finite_members_hist"], {"17": 1, "22": 2})
        self.assertEqual(Q[2].tobytes(), P[2].tobytes())
        self.assertEqual(info["status_counts"]["ok"], 1)
        r = self._run_stage(nan_members=4)                              # 18 of 22 finite: edited
        self.assertEqual(r["info"]["too_few_members_frames"], 0)
        self.assertEqual(r["info"]["status_counts"]["ok"], 2)

    def test_stale_kappa_falls_back(self):
        """59e: a kappa meta from another run (run id), other inputs (fingerprint) or another ensemble (SHA-256) makes
        d3d_mmk fall back, and the submission names the fallback."""
        for k, v, why in (("run_id", "r0", "run_id"), ("inputs_sha256", "0" * 64, "inputs_sha256"),
                          ("ensemble_psi_sha256", "1" * 64, "ensemble_psi_sha256")):
            r = self._run_stage(tamper=lambda m, k=k, v=v: m.__setitem__(k, v))
            self.assertEqual(r["info"]["fallback"], "src", k)
            self.assertIn(why, r["info"]["error"])
            self.assertIsNone(r["Q"])
            self.assertEqual(r["submit_marker"], "mmk_src")

    def test_worker_failure_in_the_cli_falls_back(self):
        """59e: the apply CLI's workers cannot start (an envelope file that is not an npz; its SHA-256 is accepted):
        the pool breaks, the CLI exits non-zero, d3d_mmk records fallback=src and writes no map."""
        with tempfile.TemporaryDirectory() as t:
            bad = Path(t) / "env_train.npz"
            bad.write_bytes(b"not an npz")
            r = self._run_stage(env=bad)
        self.assertEqual(r["info"]["fallback"], "src")
        self.assertIn("exit code", r["info"]["error"])
        self.assertIsNone(r["Q"])
        self.assertEqual(r["submit_marker"], "mmk_src")

    def test_stage_falls_back_without_member_kappa(self):
        """No kappa file (d3d_b's hook failed or did not run): d3d_mmk records fallback=src, lirep4 reads the dec preds."""
        import fec.run_final as RF
        rec = json.loads(json.dumps(RF.RECIPES["final_MTfull_mmk"]))
        with tempfile.TemporaryDirectory() as d:
            c = RF.Ctx(_args(root=d, dry_run=False), rec)
            c.L.manifest_dir.mkdir(parents=True, exist_ok=True)
            for n in ("d3d_q95h", "d3d_dec"):
                (c.L.manifest_dir / f"{n}.json").write_text(json.dumps(dict(status="ok", info=dict(fallback=None))))
            src, out = RF.mmk_dirs(c)
            _write_src(src, np.stack([solovev()]))
            for f in ("cfg_psi.npy", "cfg_scal.npy"):
                (src / f).rename(src / f.replace("cfg", c.L.d3d_cfg))
            with mock.patch.object(RF, "mmk_env", lambda c: (_Env.path, {})), \
                    mock.patch.object(RF, "d3d_pred_N", lambda c: 1):
                info = RF.st_d3d_mmk(c)
            self.assertEqual(info["fallback"], "src")
            self.assertIn("missing", info["error"])
            self.assertFalse((out / f"{c.L.d3d_cfg}_psi.npy").exists())


class TestS1MmkConfig(unittest.TestCase):
    """final_A22li_dec_mmk (agent mmk2pkg 2026-10-20) = final_A22li_dec + the same d3d.mmk block as final_MTfull_mmk."""

    def test_recipe_is_a22li_dec_plus_the_same_mmk(self):
        import fec.run_final as RF
        a, b = RF.RECIPES["final_A22li_dec_mmk"], RF.RECIPES["final_A22li_dec"]
        a2 = json.loads(json.dumps(a))
        q = a2["d3d"].pop("mmk")
        a2.pop("name")
        b2 = json.loads(json.dumps(b))
        b2.pop("name")
        self.assertEqual(a2, b2)
        self.assertEqual(q, RF.RECIPES["final_MTfull_mmk"]["d3d"]["mmk"])
        self.assertEqual(q, RF.MMK_KAPPA)
        self.assertEqual(MMK.check_frozen(q), [])
        self.assertIn("_dec_mmk_lirep4__", a["name"])
        self.assertNotIn("mmk", b["d3d"])
        st = RF.all_stages(a)
        i = st.index("d3d_q95h")
        self.assertEqual(st[i:i + 4], ["d3d_q95h", "d3d_dec", "d3d_mmk", "d3d_lirep4"])
        self.assertEqual([x for x in st if x != "d3d_mmk"], RF.all_stages(b))

    def test_dry_run_commands_and_members(self):
        import fec.run_final as RF
        stages = ["d3d_b", "d3d_q95h", "d3d_dec", "d3d_lirep4"]
        with tempfile.TemporaryDirectory() as d:
            mt = TestRunFinalWiring._cmds(None, "final_A22li_dec", stages, d)
            mk = TestRunFinalWiring._cmds(None, "final_A22li_dec_mmk", stages[:3] + ["d3d_mmk"] + stages[3:], d)
            c = RF.Ctx(_args(root=d), json.loads(json.dumps(RF.RECIPES["final_A22li_dec_mmk"])))
            tags = RF.mmk_recipe_tags(c)
        self.assertFalse(any("--mmk" in x or "mmk.py" in x or "_mmk" in x for x in mt))
        self.assertEqual(mk[0].split(" --mmk-env ")[0].replace(d, ""), mt[0].replace(d, ""))
        self.assertTrue(mk[0].endswith("--mmk-nproc 5 --mmk-run-id dry-run --mmk-stall-timeout 600.0"), mk[0])
        self.assertEqual([x.replace(d, "") for x in mk[1:3]], [x.replace(d, "") for x in mt[1:3]])
        for k in ("mmk.py apply", "d3d_ip_C2_w3k22liA_q95h_dec_mmk", "--min-members 18", "--operator mmk_op",
                  "--stall-timeout 600.0", "--skip-rows"):
            self.assertIn(k, mk[3])
        kap = Path(mk[3].split("--kappa ")[1].split(" ")[0])
        self.assertEqual((kap.parent.name, kap.name), ("d3d_ip_C2_w3k22liA", "diii_d_private_test_mmk_kappa.npy"))
        self.assertIn("d3d_ip_C2_w3k22liA_q95h_dec_mmk", mk[4])
        self.assertEqual(len(tags), 22)
        self.assertTrue(all(t.startswith("d3d_li2ftA22_") for t in tags), tags)


# ---------------------------------------------------------------------------------------------- 59e: pools and faults
def _ok(x):
    return x * 2


def _boom(x):
    raise ValueError(f"boom {x}")


def _die(x):
    os._exit(3)


def _hang(x):
    time.sleep(120)
    return x


def _die_task(args):            # a member-kappa / apply task whose worker dies
    os._exit(3)


def _hang_task(args):           # a member-kappa / apply task that never finishes in time
    time.sleep(120)


def _pool(n=1):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))


class TestPools(unittest.TestCase):
    def test_results_complete_and_bounded(self):
        ex = _pool(2)
        try:
            got = sorted(MMK.bounded_map(ex, _ok, range(20), depth=3, stall_timeout=60))
        finally:
            ex.shutdown(wait=True)
        self.assertEqual(got, [2 * i for i in range(20)])

    def test_worker_error_dead_worker_and_stall_raise(self):
        from concurrent.futures.process import BrokenProcessPool
        # a long stall limit where the expected failure is not a stall (worker start-up can take long on a loaded host)
        for fn, exc, stall in ((_boom, ValueError, 120), (_die, BrokenProcessPool, 120), (_hang, MMK.PoolStall, 5)):
            ex = _pool(1)
            t0 = time.time()
            try:
                with self.assertRaises(exc, msg=fn.__name__):
                    list(MMK.bounded_map(ex, fn, range(3), depth=2, stall_timeout=stall))
            finally:
                procs = list((getattr(ex, "_processes", None) or {}).values())
                MMK.kill_pool(ex)
            self.assertLess(time.time() - t0, 150, fn.__name__)
            for p in procs:                                           # a hung worker is really ended
                p.join(timeout=20)
                self.assertFalse(p.is_alive(), fn.__name__)


class _DieMK(MMK.MemberKappa):
    task = staticmethod(_die_task)


class _HangMK(MMK.MemberKappa):
    task = staticmethod(_hang_task)

    def __init__(self, *a, **kw):
        kw["stall_timeout"] = 5
        super().__init__(*a, **kw)


class _SaveFailMK(MMK.MemberKappa):
    def save(self, path, extra=None):
        Path(path).write_bytes(b"partial")                            # a half-written file must not survive
        raise OSError("disk full")


class TestHookFailures(unittest.TestCase):
    """59e (review F2 / F3): every failure of the member-kappa part inside predict_b_chunked leaves the ensemble files
    byte-identical to a run without the hook, writes no kappa file and removes an earlier run's files."""
    KAP = TestMemberKappaHook.KAP
    _member = TestMemberKappaHook._member

    @classmethod
    def setUpClass(cls):
        import fec.run_final as RF
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.t = Path(cls.tmp.name)
        cls.rec = t / "members.json"
        cls.rec.write_text(json.dumps({"members": [{"kind": "b", "tag": x, "weight": 1.0} for x in ("m_a", "m_b", "m_c")],
                                       "fallback": [{"kind": "b", "tag": "fb", "weight": 1.0}]}))
        cls.nots = np.zeros(6, bool)
        cls.nots[2] = True
        with cls._patches():
            RF.predict_b_chunked("cfg", t / "plain", cls.rec, chunk_rows=4096, log=lambda m: None)
        cls.plain = {f: _sha(t / "plain" / f) for f in ("cfg_psi.npy", "cfg_scal.npy")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def _patches(cls, *extra):
        import contextlib
        st = contextlib.ExitStack()
        st.enter_context(mock.patch("fec.predict_d3d_b._member", TestMemberKappaHook._member))
        st.enter_context(mock.patch("fec.features.load_ds", lambda cfg: {"info": {"N": 6}}))
        st.enter_context(mock.patch("fec.features_b.no_thomson_frames", lambda ds: cls.nots))
        for e in extra:
            st.enter_context(e)
        return st

    def _run(self, name, env=None, extra=(), stale=False):
        import fec.run_final as RF
        out = self.t / name
        if stale:                                                     # an earlier run's kappa files
            out.mkdir()
            for f in ("cfg_mmk_kappa.npy", "cfg_mmk_kappa.json", "cfg_mmk_fallback_rows.npy"):
                (out / f).write_bytes(b"old")
        logs = []
        t0 = time.time()
        with self._patches(*extra):
            RF.predict_b_chunked("cfg", out, self.rec, chunk_rows=4096, log=logs.append,
                                 mmk=dict(env=str(env or _Env.path), nproc=1, run_id="r9", stall_timeout=120))
        self.assertLess(time.time() - t0, 300, name)
        for f, h in self.plain.items():
            self.assertEqual(_sha(out / f), h, (name, f))
        for f in ("cfg_mmk_kappa.npy", "cfg_mmk_kappa.json", "cfg_mmk_fallback_rows.npy"):
            self.assertFalse((out / f).exists(), (name, f))
        meta = json.loads((out / "cfg_meta.json").read_text())
        self.assertTrue(any("FAILED" in m for m in logs), name)
        return meta["mmk_kappa"]["error"]

    def test_setup_and_import_errors(self):
        self.assertTrue(self._run("nofile", env=self.t / "missing_env.npz", stale=True).startswith("setup: "))
        e = self._run("noimport", extra=(mock.patch.dict(sys.modules, {"fec.mmk": None}),))
        self.assertTrue(e.startswith("setup: ") and ("ImportError" in e or "ModuleNotFoundError" in e), e)

    def test_initializer_error(self):
        bad = self.t / "garbage_env.npz"
        bad.write_bytes(b"not an npz")
        e = self._run("badenv", env=bad)
        self.assertTrue(e.startswith("member kappa: ") and "BrokenProcessPool" in e, e)

    def test_dead_and_stalled_workers(self):
        e = self._run("dead", extra=(mock.patch("fec.mmk.MemberKappa", _DieMK),), stale=True)
        self.assertTrue(e.startswith("member kappa: ") and "BrokenProcessPool" in e, e)
        e = self._run("stall", extra=(mock.patch("fec.mmk.MemberKappa", _HangMK),))
        self.assertTrue(e.startswith("member kappa: ") and "PoolStall" in e, e)

    def test_close_save_error(self):
        e = self._run("save", extra=(mock.patch("fec.mmk.MemberKappa", _SaveFailMK),))
        self.assertTrue(e.startswith("close/save: ") and "disk full" in e, e)

    def test_success_records_run_id_inputs_and_ensemble(self):
        import fec.run_final as RF
        out = self.t / "good"
        with self._patches():
            RF.predict_b_chunked("cfg", out, self.rec, chunk_rows=4096, log=lambda m: None,
                                 mmk=dict(env=str(_Env.path), nproc=1, run_id="r9", stall_timeout=60))
        km = json.loads((out / "cfg_mmk_kappa.json").read_text())
        self.assertEqual(km["run_id"], "r9")
        self.assertEqual(km["ensemble_psi_sha256"], self.plain["cfg_psi.npy"])
        self.assertEqual(km["ensemble_scal_sha256"], self.plain["cfg_scal.npy"])
        self.assertEqual(km["inputs"]["tags"], ["m_a", "m_b", "m_c"])
        self.assertEqual(km["inputs"]["N"], 6)
        _, ish = RF.mmk_inputs("cfg", self.rec, 6, ["m_a", "m_b", "m_c"], Path(RF.WORK) / "ds" / "cfg" / "info.json")
        self.assertEqual(km["inputs_sha256"], ish)
        self.assertEqual(km["finite_members_min"], 3)


class TestApplyFailuresAndMinMembers(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        f = _func()
        self.P = np.stack([solovev(kap=k) for k in (1.70, 1.66, 1.74, 1.72)])
        K = np.array([[_kappa(f, p) + 0.02 + 0.001 * m for m in range(5)] for p in self.P])
        K[1, :2] = np.nan                                             # 3 of 5 finite
        K[3, :3] = np.nan                                             # 2 of 5 finite, and a fallback row
        np.save(self.d / "k.npy", K)
        np.save(self.d / "s.npy", np.array([3], np.int64))
        self.src = _write_src(self.d / "src", self.P)

    def tearDown(self):
        self.tmp.cleanup()

    def test_min_members(self):
        m = MMK.apply(self.src, self.d / "o", "cfg", self.d / "k.npy", _Env.path, nproc=1, skip_rows=self.d / "s.npy",
                      min_members=4, log=lambda x: None)
        st = np.load(self.d / "o" / "cfg_mmk_flags.npz")["k_status"]
        self.assertEqual(st[1], MMK.TOO_FEW_MEMBERS)
        self.assertEqual(st[3], MMK.FALLBACK_FRAME)                    # the fallback rule comes first
        self.assertEqual((m["too_few_members_frames"], m["fallback_frames_not_edited"]), (1, 1))
        Q = np.load(self.d / "o" / "cfg_psi.npy")
        for i in (1, 3):
            self.assertEqual(Q[i].tobytes(), self.P[i].tobytes())
        self.assertEqual(m["finite_members_hist"], {"2": 1, "3": 1, "5": 2})
        m0 = MMK.apply(self.src, self.d / "o0", "cfg", self.d / "k.npy", _Env.path, nproc=1,
                       skip_rows=self.d / "s.npy", min_members=0, log=lambda x: None)
        self.assertEqual(m0["too_few_members_frames"], 0)               # 0 = no minimum (confirm1's object)
        with self.assertRaises(SystemExit):
            MMK.apply(self.src, self.d / "o6", "cfg", self.d / "k.npy", _Env.path, nproc=1, min_members=6,
                      log=lambda x: None)

    def test_dead_or_stalled_worker_writes_no_map(self):
        from concurrent.futures.process import BrokenProcessPool
        for name, task, exc, stall in (("dead", _die_task, BrokenProcessPool, 120), ("stall", _hang_task, MMK.PoolStall, 5)):
            t0 = time.time()
            with self.assertRaises(exc):
                MMK.apply(self.src, self.d / name, "cfg", self.d / "k.npy", _Env.path, nproc=1, stall_timeout=stall,
                          task=task, log=lambda x: None)
            self.assertLess(time.time() - t0, 180)
            self.assertFalse((self.d / name / "cfg_psi.npy").exists())
            self.assertFalse((self.d / name / "cfg_psi.part.npy").exists())


class TestOperatorRegistry(unittest.TestCase):
    def test_registered_operators_are_alpha_only_variants(self):
        for name, h in MMK.OPERATORS.items():
            self.assertEqual(_sha(REPO / "fec" / f"{name}.py"), h, name)
            self.assertTrue(MMK.alpha_only_variant(name), name)
            self.assertEqual(MMK.check_frozen(op_name=name), [], name)
        self.assertEqual(MMK.OPERATORS["mmk_op"], MMK.OP_SHA256)

    def test_make_operator_and_checks(self):
        import fec.run_final as RF
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "mmk_op_a075.py"
            h = MMK.make_operator(0.75, out)
            a = (REPO / "fec" / "mmk_op.py").read_bytes().split(b"\n")
            b = out.read_bytes().split(b"\n")
            diff = [(x, y) for x, y in zip(a, b) if x != y]
            self.assertEqual((len(a), diff), (len(b), [(b"ALPHA = 0.5", b"ALPHA = 0.75")]))
            self.assertNotEqual(h, MMK.OP_SHA256)
            self.assertEqual(h, _sha(out))
            with self.assertRaises(ValueError):
                MMK.make_operator(0.5, Path(t) / "x.py")
        self.assertTrue(MMK.check_frozen(op_name="mmk_op_a075"))        # not registered
        q = dict(RF.MMK_KAPPA)
        self.assertEqual(MMK.check_frozen(q), [])
        self.assertTrue(MMK.check_frozen(dict(q, alpha=0.75)))         # the recipe's alpha must be the file's
        self.assertTrue(MMK.check_frozen(dict(q, operator="fec/mmk_op_a075.py")))

    def test_docs_state_the_recipes_alpha(self):
        """The README paragraph on the kappa finishing states the recipe's alpha (update both if 59a changes it)."""
        import fec.run_final as RF
        alpha = RF.RECIPES["final_MTfull_mmk"]["d3d"]["mmk"]["alpha"]
        for f in (REPO / "README.md", REPO / "fec" / "final_pkg_tmpl" / "README.md"):
            if f.is_file():
                t = " ".join(f.read_text(encoding="utf-8").split())
                if "Member-mean κ finishing" in t:
                    para = t.split("**Member-mean κ finishing", 1)[1][:3000]
                    self.assertIn(f"(α = {alpha:g})", para)


if __name__ == "__main__":
    unittest.main()
