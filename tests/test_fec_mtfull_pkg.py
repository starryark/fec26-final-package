"""final_MTfull = final_A22li_dec with the 22 fine-tuned members replaced by their "MT" twins (mtask M2 auxiliary heads
on the bases, li2ft A with the heads kept on; agent mtpkg 2026-10-15; family doc amendment 39; EVIDENCE B75;
docs/archive/fec/FINAL_MTFULL_PKG_20261015.md).

Shipped in the Final package (tests/) and run in the research repo (a temporary package skeleton written by
fec/final_pkg.py).  CPU only, config-level or synthetic:
  * config = final_A22li_dec except the DIII-D ensemble name / recipe (same decoder block, same r = 2 MAST half);
  * the members recipe = final_A22li's with the 22 li2ft tags renamed (weights and the old-28 fallback unchanged);
  * train spec: MT bases F1 / F2, the old-28 groups, MT fine-tunes F1 / F2; group size 4 (the production stacks); the
    job lines are final_A22li's (w3k22_F1 / F2a + F2b / li2ftA22) with the tags renamed and nothing else;
  * train.py: li2 cache -> aux cache -> members -> decoders -> mast; aux flags on the four MT groups only; the aux
    stage is a silent no-op for every other config; per-config disk peak;
  * fec/mtask_aux_all.py maps every row to its li2-cache position (synthetic worker);
  * builder: the MT bases get no weights_manifest row, the MT fine-tunes do; every other group's rows are unchanged;
  * final_A24r2 (amendment 46): final_A24's DIII-D half and train spec exactly, the r = 2 MAST composite, in
    package_cands; final_A24 itself unchanged.
  * final_MTli (agent mtlipkg 2026-10-16; amendment 50 (A); EVIDENCE B71 file 728f71e1): final_MTfull's F1 (MT bases and
    MT fine-tunes, the same groups and lines) + final_A22li_dec's F2 (the W3k22 no-Thomson bases and their plain li2ft
    lines, li2ftA22_F2 = the no-Thomson half of li2ftA22 verbatim) + the same old-28 fallback; the same decoder block
    and r = 2 MAST node; no new weight row.
Research repo only (skipped elsewhere): the all-row aux cache built locally equals production's byte for byte; three
STORED local runs of the package inference path (2026-10-15 and 2026-10-16) reproduced the Dev files: final_MTfull
(DIII-D 68f168a6, MAST 12e264e5), final_A24r2 (MAST 12e264e5) and final_MTli (728f71e1, MAST 12e264e5).  Those runs used
the Development-phase MAST net (model.pt 529ebce2, the net of weights revision 95f88c25) installed in the local work
dir, so these tests check those stored outputs and that net's file 12e264e5.  They do not check the MAST net this
package ships at weights revision dfa0ee10 (the A100 retrain bfbb346c; README "Expected rebuild variance"), whose maps differ from 12e264e5.
"""
import hashlib
import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CFG = "final_MTfull"
MT_GROUPS = ["mt22_F1", "mt22_F2", "old_notsL_axij", "old_notsL_axkl", "mtftA22_F1", "mtftA22_F2"]
AUX_ARGS = ["--aux_heads", "shape_jphi", "--aux_w", "1.0"]
# Dev files of the MT-full read (EVIDENCE B75, Codabench 988521) and the clean r = 2 MAST file (C66, 987175).  DEV_MAST_SHA
# is the Development-phase MAST net's file (weights revision 95f88c25), not the shipped net's (dfa0ee10; README "Expected rebuild variance").
DEV_D3D_SHA = "68f168a6e4930841ef568b5fd3ac9873efbd5ad132a64a576489478083d9c28c"
DEV_MAST_SHA = "12e264e51e931801f8f0beaec5d9df811f870fe7a940f0f7b72f7531cd3b04f0"
PROD_AUX_MD5 = ("93fe30ea", "581929b9")             # fec_work/mtprod/aux_all aux.npy / gidx.npy (D3D_MTPROD section 1)
# final_MTli (amendment 50 (A)): its groups, its members recipe (byte copy of fec_work/mtprod/final_ip_C2_w3k22liAmt.json,
# sha256 of the LF bytes) and its public-test DIII-D file (EVIDENCE B71; built by fec/pod/mtprod/build_mt.sh)
MTLI = "final_MTli"
MTLI_GROUPS = ["mt22_F1", "w3k22_F2a", "w3k22_F2b", "old_notsL_axij", "old_notsL_axkl", "mtftA22_F1", "li2ftA22_F2"]
MTLI_RECIPE_SHA_LF = "568dc572a339a49437ea6859dbcde1c3c87f24f16f8897908e52a910fd4926d5"
MTLI_D3D_SHA = "728f71e1464c07f2d753306be52627ae31f43a1775c0f769e7bee202ced7a206"


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).parent))
    spec.loader.exec_module(m)
    return m


def _rename(line):
    return line.replace("d3d_bxw3e_", "d3d_bxw3emt_").replace("d3d_li2ftA22_", "d3d_li2ftA22mt_")


def _tag(line):
    t = shlex.split(line)
    return t[t.index("--tag") + 1]


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
        cls.groups = FP.train_configs(pkg)
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

    def spec(self, name=CFG):
        return json.loads((self.pkg / "configs" / f"train_d3d_{name}.json").read_text())

    def jobs(self, group):
        return json.loads((self.pkg / "configs" / "jobs" / f"{group}.json").read_text())

    def _dry(self, cfg, stage="all"):
        with tempfile.TemporaryDirectory() as w:
            p = subprocess.run([sys.executable, "-B", str(self.pkg / "train.py"), "--config", cfg, "--stage", stage,
                                "--dry-run", "--work", w], cwd=str(self.pkg), capture_output=True, text=True,
                               timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        return p.stdout, [ln[2:] for ln in p.stdout.splitlines() if ln.startswith("$ ")]


class TestMTfullConfig(_Pkg):
    def test_config_is_a22li_dec_with_the_mt_ensemble(self):
        mt, dec = self.cfg(), self.cfg("final_A22li_dec")
        self.assertEqual(mt["d3d"]["b"].pop("name"), "d3d_ip_C2_w3k22liAmtfull")
        self.assertEqual(mt["d3d"]["b"].pop("recipe"), "fec/recipes_final/final_ip_C2_w3k22liAmtfull.json")
        self.assertEqual(dec["d3d"]["b"].pop("name"), "d3d_ip_C2_w3k22liA")
        dec["d3d"]["b"].pop("recipe")
        self.assertIn("w3k22liAmtfull", mt.pop("name"))
        dec.pop("name")
        self.assertEqual(mt, dec)                      # same dec block (tag, dir, shipped basis), lirep4, r = 2 MAST
        self.assertEqual(mt["d3d"]["dec"]["tag"], "offdec_d3d_ip_C2_w3k22liA")
        self.assertEqual(mt["mast"]["composite"]["maps"]["rscale"], 2)

    def test_members_recipe_is_a22lis_with_the_mt_tags(self):
        mt = json.loads((self.pkg / "fec" / "recipes_final" / "final_ip_C2_w3k22liAmtfull.json").read_text())
        a = json.loads((self.pkg / "fec" / "recipes_final" / "final_ip_C2_w3k22liA.json").read_text())
        self.assertEqual(mt["fallback"], a["fallback"])
        self.assertEqual(len(mt["members"]), 22)
        self.assertEqual([dict(m, tag=_rename(m["tag"])) for m in a["members"]], mt["members"])
        self.assertTrue(all(m["tag"].startswith("d3d_li2ftA22mt_") for m in mt["members"]))
        self.assertAlmostEqual(sum(m["weight"] for m in mt["members"]), 1.0, places=12)

    def test_train_spec(self):
        s = self.spec()
        self.assertEqual([g["name"] for g in s["groups"]], MT_GROUPS)
        n = {g["name"]: g["n"] for g in s["groups"]}
        self.assertEqual(n, {"mt22_F1": 11, "mt22_F2": 11, "old_notsL_axij": 14, "old_notsL_axkl": 14,
                             "mtftA22_F1": 11, "mtftA22_F2": 11})
        for g in s["groups"]:
            mt = g["name"].startswith("mt")
            self.assertEqual(g["group_size"], 4 if mt else 7, g["name"])
            self.assertEqual(g.get("aux"), {"heads": "shape_jphi", "w": 1.0} if mt else None, g["name"])
            self.assertEqual(bool(g.get("li2_cache")), g["name"].startswith("mtft"), g["name"])
            self.assertEqual(g["tag_replace"], ["d3d_b60ax=d3d_bxax"] if g["name"] == "old_notsL_axkl" else [])
        old = {g["name"]: g for g in self.spec("final_A22li_dec")["groups"]}
        for name in ("old_notsL_axij", "old_notsL_axkl"):          # the fallback groups are final_A22li_dec's
            self.assertEqual([g for g in s["groups"] if g["name"] == name][0], old[name])
        for name in old:                                           # no aux key leaks into the other configs
            self.assertNotIn("aux", old[name])

    def test_job_lines_are_a22lis_renamed(self):
        """MT bases = final_A22li's W3k22 base lines, MT fine-tunes = its li2ftA22 lines (--ft_from included), tags
        renamed only; job order = the production stacks."""
        self.assertEqual(self.jobs("mt22_F1"), [_rename(x) for x in self.jobs("w3k22_F1")])
        self.assertEqual(self.jobs("mt22_F2"), [_rename(x) for x in self.jobs("w3k22_F2a") + self.jobs("w3k22_F2b")])
        li = self.jobs("li2ftA22")
        self.assertEqual(self.jobs("mtftA22_F1") + self.jobs("mtftA22_F2"), [_rename(x) for x in li])
        f1 = [_tag(x).split("_s")[-1] for x in self.jobs("mt22_F1")]
        f2 = [_tag(x).split("_s")[-1] for x in self.jobs("mt22_F2")]
        self.assertEqual(f1, ["0", "1", "2", "3", "40", "41", "42", "43", "80", "81", "82"])
        self.assertEqual(f2, ["5", "6", "7", "45", "46", "47", "85", "86", "87", "125", "126"])
        for g, b in (("mtftA22_F1", "mt22_F1"), ("mtftA22_F2", "mt22_F2")):
            ft = [shlex.split(x) for x in self.jobs(g)]
            self.assertEqual([t[t.index("--ft_from") + 1] for t in ft], [_tag(x) for x in self.jobs(b)])
        for x in self.jobs("mt22_F1") + self.jobs("mt22_F2") + li:
            self.assertIn("--alldata", shlex.split(x))

    def test_train_order_aux_flags_and_caches(self):
        _, cmds = self._dry(CFG)
        li2 = [i for i, c in enumerate(cmds) if "li2_torch.py" in c]
        aux = [i for i, c in enumerate(cmds) if "mtask_aux" in c]
        bx = [i for i, c in enumerate(cmds) if "train_d3d_bx.py" in c]
        od = [i for i, c in enumerate(cmds) if "offdec.py" in c]
        mast = [i for i, c in enumerate(cmds) if "final_mast.py" in c or "amort_net2.py" in c or "amort_gen7" in c]
        self.assertEqual((len(li2), len(aux), len(bx), len(od), len(mast)), (2, 2, 6, 6, 3))
        self.assertTrue(max(li2) < min(aux) and max(aux) < min(bx) and max(bx) < min(od) and max(od) < min(mast))
        self.assertIn("mtask_aux_all.py build", cmds[aux[0]])
        self.assertIn("mtask_aux.py check", cmds[aux[1]])
        sep = "\\" if "\\" in cmds[aux[0]] else "/"
        auxdir = sep.join(["fec_work_idx", "mtask", "aux_all"])
        li2dir = sep.join(["fec_work_idx", "li2", "cache_all"])
        self.assertIn(auxdir, cmds[aux[0]])
        self.assertIn(li2dir, cmds[aux[0]])
        names = [c.split("--jobs ")[1].split()[0].replace("\\", "/").rsplit("/", 1)[1][:-5] for c in (cmds[i] for i in bx)]
        self.assertEqual(names, MT_GROUPS)
        for nm, i in zip(names, bx):
            toks = cmds[i].split()
            has_aux = all(a in toks for a in AUX_ARGS) and auxdir in cmds[i]
            self.assertEqual(has_aux, nm.startswith("mt"), nm)
            self.assertEqual("--li2_cache" in toks, nm.startswith("mtft"), nm)
            if nm.startswith("mt"):
                self.assertEqual(toks[toks.index("--group_size") + 1], "4")
        self.assertIn("final_ip_C2_w3k22liAmtfull.json", cmds[od[0]])
        self.assertIn(sep.join(["offdec", "d3d_ip_C2_w3k22liA", "dec_s40.pt"]), cmds[od[3]])
        # --stage d3d runs the caches first, then members, then the decoders (reproduce.sh full: data, d3d, mast, export)
        _, c2 = self._dry(CFG, "d3d")
        self.assertEqual([c.split(".py")[0].replace("\\", "/").rsplit("/", 1)[1] for c in c2 if "train_d3d_bx" not in c
                          and "offdec" not in c], ["li2_torch", "li2_torch", "mtask_aux_all", "mtask_aux"])

    def test_auxcache_stage_alone(self):
        out, cmds = self._dry(CFG, "auxcache")
        self.assertEqual([c.split(".py")[0].replace("\\", "/").rsplit("/", 1)[1] for c in cmds],
                         ["li2_torch", "li2_torch", "mtask_aux_all", "mtask_aux"])
        self.assertIn("auxcache: for groups ['mt22_F1', 'mt22_F2', 'mtftA22_F1', 'mtftA22_F2']", out)
        for other in ("final_A22li_dec", "final_A22li", "final_A24", "final_B"):
            if not (self.pkg / "configs" / f"{other}.json").exists():
                continue
            out, c2 = self._dry(other, "auxcache")
            self.assertIn("has no aux group; nothing to build", out, other)
            self.assertFalse(any("mtask_aux" in c for c in c2), other)

    def test_no_aux_anywhere_else(self):
        for other in ("final_A22li_dec", "final_A22li", "final_A24", "final_B", "final_C"):
            if not (self.pkg / "configs" / f"{other}.json").exists():
                continue
            out, cmds = self._dry(other)
            self.assertFalse(any("mtask_aux" in c or "--aux_heads" in c or "--aux_cache" in c for c in cmds), other)
            self.assertNotIn("auxcache", out, other)

    def test_peak_disk_and_tags(self):
        T = _module(self.pkg / "train.py", "pkg_train_mtfull_test")
        self.assertEqual(T.peak_disk_gb(CFG), T.PEAK_DISK_GB + 1)
        self.assertEqual(T.peak_disk_gb("final_A22li_dec"), T.PEAK_DISK_GB + 1)
        self.assertEqual(T.peak_disk_gb("final_A22li"), T.PEAK_DISK_GB)
        self.assertEqual(T.aux_groups(CFG), ["mt22_F1", "mt22_F2", "mtftA22_F1", "mtftA22_F2"])
        self.assertEqual(T.aux_groups("final_A22li_dec"), [])
        I = _module(self.pkg / "infer.py", "pkg_infer_mtfull_test")
        tags = I.recipe_tags(self.cfg())
        self.assertEqual(len(tags), 50)
        self.assertEqual(len([t for t in tags if t.startswith("d3d_li2ftA22mt_")]), 22)
        self.assertEqual(I.dec_tags(self.cfg()), I.dec_tags(self.cfg("final_A22li_dec")))
        a22 = I.recipe_tags(self.cfg("final_A22li_dec"))
        self.assertEqual({_rename(t) for t in a22}, tags)

    def test_weights_manifest_rows(self):
        """Inside a built package: the 22 MT fine-tunes have rows, the MT bases have none, decoder rows stay 3."""
        f = REPO / "weights_manifest.json"
        if not (REPO / "configs" / f"{CFG}.json").exists() or not f.exists():
            self.skipTest("weights_manifest.json is written by the package build")
        rows = json.loads(f.read_text())["files"]
        d3d = {r["tag"] for r in rows if r.get("group") == "d3d"}
        self.assertEqual(len([t for t in d3d if t.startswith("d3d_li2ftA22mt_")]), 22)
        self.assertFalse([t for t in d3d if t.startswith("d3d_bxw3emt_")])
        self.assertEqual(len([r for r in rows if r.get("group") == "offdec"]), 3)
        for r in rows:
            if r.get("tag", "").startswith("d3d_li2ftA22mt_"):
                self.assertRegex(r["sha256"] or "", r"^[0-9a-f]{64}$")


class TestA24r2Config(_Pkg):
    """final_A24r2 (agent mtpkg 2026-10-15; amendment 46 ruling after rev_div): final_A24's DIII-D half exactly, the r = 2
    MAST composite of final_A22li_dec / final_MTfull; final_A24 itself unchanged (r = 1 maps)."""

    def test_d3d_half_is_final_a24s(self):
        a, b = self.cfg("final_A24r2"), self.cfg("final_A24")
        self.assertEqual(a["d3d"], b["d3d"])
        self.assertNotIn("dec", a["d3d"])
        self.assertEqual(self.spec("final_A24r2")["groups"], self.spec("final_A24")["groups"])
        for k in ("psi_dtype", "train_ranges"):
            self.assertEqual(a.get(k), b.get(k), k)

    def test_mast_half_is_the_r2_composite(self):
        a, b, mt, dec = (self.cfg(n) for n in ("final_A24r2", "final_A24", "final_MTfull", "final_A22li_dec"))
        self.assertEqual(a["mast"]["composite"]["maps"]["rscale"], 2)
        self.assertEqual(a["mast"], mt["mast"])                    # the same MAST node as final_MTfull / A22li_dec
        self.assertEqual(a["mast"], dec["mast"])
        self.assertNotIn("rscale", b["mast"]["composite"]["maps"])   # final_A24 keeps r = 1
        self.assertEqual(dict(b["mast"]["composite"]["maps"], rscale=2), a["mast"]["composite"]["maps"])
        a.pop("name"); b.pop("name")
        a["mast"]["composite"]["maps"].pop("rscale")
        self.assertEqual(a, b)                                     # nothing else differs from final_A24

    def test_train_and_infer_listing(self):
        """Training commands = final_A24's (no decoder, no aux, no li2 cache); same weight tags."""
        with tempfile.TemporaryDirectory() as w:                  # one --work for both: commands comparable verbatim
            out = []
            for c in ("final_A24r2", "final_A24"):
                p = subprocess.run([sys.executable, "-B", str(self.pkg / "train.py"), "--config", c, "--stage", "all",
                                    "--dry-run", "--work", w], cwd=str(self.pkg), capture_output=True, text=True,
                                   timeout=600)
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
                out.append([ln for ln in p.stdout.splitlines() if ln.startswith("$ ")])
        self.assertEqual(out[0], out[1])
        self.assertFalse(any("offdec.py" in c or "mtask_aux" in c or "li2_torch" in c for c in out[0]))
        I = _module(self.pkg / "infer.py", "pkg_infer_a24r2_test")
        self.assertEqual(I.recipe_tags(self.cfg("final_A24r2")), I.recipe_tags(self.cfg("final_A24")))
        self.assertEqual(I.dec_tags(self.cfg("final_A24r2")), set())
        self.assertEqual(I.mast_tags(self.cfg("final_A24r2")), I.mast_tags(self.cfg("final_A24")))


class TestMTliConfig(_Pkg):
    """final_MTli (agent mtlipkg 2026-10-16; family doc amendment 50 (A)): final_MTfull's F1 half + final_A22li_dec's F2
    half and old-28 fallback; q95h -> dec (DEC_H1DEC, the decoders 728f71e1 used) -> lirep4; the r = 2 MAST node."""

    def recipe(self, name):
        return json.loads((self.pkg / "fec" / "recipes_final" / name).read_text())

    def test_config_is_a22li_dec_and_mtfull_except_the_ensemble(self):
        mt, dec, full = self.cfg(MTLI), self.cfg("final_A22li_dec"), self.cfg("final_MTfull")
        self.assertEqual(mt["d3d"]["b"].pop("name"), "d3d_ip_C2_w3k22liAmt")
        self.assertEqual(mt["d3d"]["b"].pop("recipe"), "fec/recipes_final/final_ip_C2_w3k22liAmt.json")
        self.assertIn("w3k22liAmt_q95hG1_dec_lirep4", mt.pop("name"))
        for c in (dec, full):
            c["d3d"]["b"].pop("name")
            c["d3d"]["b"].pop("recipe")
            c.pop("name")
        self.assertEqual(mt, dec)                      # same chain, dec block, lirep4, MAST, train ranges
        self.assertEqual(mt, full)
        self.assertEqual(mt["d3d"]["dec"]["tag"], "offdec_d3d_ip_C2_w3k22liA")
        self.assertEqual(mt["d3d"]["dec"]["seeds"], [40, 41, 42])

    def test_mast_node_is_the_r2_composite(self):
        mt = self.cfg(MTLI)
        self.assertEqual(mt["mast"]["composite"]["maps"]["rscale"], 2)
        for other in ("final_MTfull", "final_A22li_dec", "final_A24r2"):
            self.assertEqual(mt["mast"], self.cfg(other)["mast"], other)

    def test_members_recipe(self):
        """F1 = final_MTfull's 11 MT fine-tunes, F2 = final_A22li_dec's 11 no-Thomson fine-tunes (each in its order,
        F1 first as in A22li), equal weights, the same old-28 fallback; the file is the mtprod build's (LF sha)."""
        f = self.pkg / "fec" / "recipes_final" / "final_ip_C2_w3k22liAmt.json"
        self.assertEqual(hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest(), MTLI_RECIPE_SHA_LF)
        mt, a, full = (self.recipe(n) for n in ("final_ip_C2_w3k22liAmt.json", "final_ip_C2_w3k22liA.json",
                                                "final_ip_C2_w3k22liAmtfull.json"))

        def half(r, fam):
            return [m for m in r["members"] if f"_{fam}" in m["tag"]]
        self.assertEqual(len(mt["members"]), 22)
        self.assertEqual(half(mt, "shape") + half(mt, "notsL"), mt["members"])
        self.assertEqual(half(mt, "shape"), half(full, "shape"))
        self.assertEqual(half(mt, "notsL"), half(a, "notsL"))
        self.assertEqual(mt["fallback"], a["fallback"])
        self.assertEqual(mt["fallback"], full["fallback"])
        self.assertEqual([dict(m, tag=m["tag"].replace("d3d_li2ftA22_shape", "d3d_li2ftA22mt_shape"))
                          for m in a["members"]], mt["members"])
        self.assertAlmostEqual(sum(m["weight"] for m in mt["members"]), 1.0, places=12)

    def test_train_spec(self):
        """Groups in order; F1 groups = final_MTfull's, F2 bases + fallback = final_A22li_dec's; li2ftA22_F2 is the
        li2 fine-tune group of 11 (li2 cache, no aux heads, group size 4)."""
        s = self.spec(MTLI)
        self.assertEqual([g["name"] for g in s["groups"]], MTLI_GROUPS)
        G = {g["name"]: g for g in s["groups"]}
        full = {g["name"]: g for g in self.spec("final_MTfull")["groups"]}
        dec = {g["name"]: g for g in self.spec("final_A22li_dec")["groups"]}
        for n in ("mt22_F1", "mtftA22_F1"):
            self.assertEqual(G[n], full[n], n)
        for n in ("w3k22_F2a", "w3k22_F2b", "old_notsL_axij", "old_notsL_axkl"):
            self.assertEqual(G[n], dec[n], n)
        li = G["li2ftA22_F2"]
        self.assertEqual((li["n"], li["group_size"], li.get("li2_cache"), li.get("aux"), li["tag_replace"]),
                         (11, 4, True, None, []))
        self.assertEqual(dec["li2ftA22"]["n"], 22)
        self.assertEqual([n for n, g in G.items() if g.get("aux")], ["mt22_F1", "mtftA22_F1"])

    def test_job_lines(self):
        """li2ftA22_F2 = li2ftA22's 11 no-Thomson lines verbatim and in order (--ft_from the w3k22_F2a / F2b bases);
        the members recipe's tags = the mtftA22_F1 + li2ftA22_F2 tags."""
        li, f2 = self.jobs("li2ftA22"), self.jobs("li2ftA22_F2")
        self.assertEqual(f2, li[11:])
        self.assertEqual(f2, [x for x in li if _tag(x).startswith("d3d_li2ftA22_notsL_")])
        self.assertEqual([_tag(x).split("_s")[-1] for x in f2],
                         ["5", "6", "7", "45", "46", "47", "85", "86", "87", "125", "126"])
        ft = [shlex.split(x) for x in f2]
        self.assertEqual([t[t.index("--ft_from") + 1] for t in ft],
                         [_tag(x) for x in self.jobs("w3k22_F2a") + self.jobs("w3k22_F2b")])
        rec = self.recipe("final_ip_C2_w3k22liAmt.json")
        self.assertEqual([m["tag"] for m in rec["members"]], [_tag(x) for x in self.jobs("mtftA22_F1") + f2])

    def test_train_dry_run_listing(self):
        """train.py --stage all --dry-run with one --work for final_MTli / final_MTfull / final_A22li_dec: li2 cache ->
        aux cache -> the 7 member groups -> decoders -> mast; each group's command is final_MTfull's (F1) or
        final_A22li_dec's (F2 bases, fallback) verbatim, li2ftA22_F2's = A22li_dec's li2ftA22 command with only the
        job file and timing names changed; caches and MAST commands = final_MTfull's, decoders = final_MTfull's with
        the members recipe renamed."""
        cmds = {}
        with tempfile.TemporaryDirectory() as w:
            for c in (MTLI, "final_MTfull", "final_A22li_dec"):
                p = subprocess.run([sys.executable, "-B", str(self.pkg / "train.py"), "--config", c, "--stage", "all",
                                    "--dry-run", "--work", w], cwd=str(self.pkg), capture_output=True, text=True,
                                   timeout=600)
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
                cmds[c] = [ln[2:] for ln in p.stdout.splitlines() if ln.startswith("$ ")]
        me = cmds[MTLI]

        def idx(cs, key):
            return [i for i, c in enumerate(cs) if key in c]

        def is_mast(c):
            return "final_mast.py" in c or "amort_net2.py" in c or "amort_gen7" in c
        li2, aux, bx, od = (idx(me, k) for k in ("li2_torch.py", "mtask_aux", "train_d3d_bx.py", "offdec.py"))
        mast = [i for i, c in enumerate(me) if is_mast(c)]
        self.assertEqual((len(li2), len(aux), len(bx), len(od), len(mast)), (2, 2, 7, 6, 3))
        self.assertTrue(max(li2) < min(aux) and max(aux) < min(bx) and max(bx) < min(od) and max(od) < min(mast))

        def group_of(c):
            return c.split("--jobs ")[1].split()[0].replace("\\", "/").rsplit("/", 1)[1][:-5]

        def by_group(cs):
            return {group_of(c): c for c in cs if "train_d3d_bx.py" in c}
        mine, full, dec = by_group(me), by_group(cmds["final_MTfull"]), by_group(cmds["final_A22li_dec"])
        self.assertEqual([group_of(me[i]) for i in bx], MTLI_GROUPS)
        for g in ("mt22_F1", "mtftA22_F1"):
            self.assertEqual(mine[g], full[g], g)
        for g in ("w3k22_F2a", "w3k22_F2b", "old_notsL_axij", "old_notsL_axkl"):
            self.assertEqual(mine[g], dec[g], g)
        self.assertEqual(mine["li2ftA22_F2"].replace("li2ftA22_F2", "li2ftA22"), dec["li2ftA22"])
        self.assertNotIn("--aux_heads", mine["li2ftA22_F2"])
        self.assertIn("--li2_cache", mine["li2ftA22_F2"])
        f = cmds["final_MTfull"]
        self.assertEqual([me[i] for i in li2 + aux], [f[i] for i in idx(f, "li2_torch.py") + idx(f, "mtask_aux")])
        self.assertEqual([me[i] for i in mast], [c for c in f if is_mast(c)])
        self.assertEqual([me[i] for i in od], [c.replace("final_ip_C2_w3k22liAmtfull.json", "final_ip_C2_w3k22liAmt.json")
                                               for c in f if "offdec.py" in c])

    def test_auxcache_stage(self):
        out, cmds = self._dry(MTLI, "auxcache")
        self.assertEqual([c.split(".py")[0].replace("\\", "/").rsplit("/", 1)[1] for c in cmds],
                         ["li2_torch", "li2_torch", "mtask_aux_all", "mtask_aux"])
        self.assertIn("auxcache: for groups ['mt22_F1', 'mtftA22_F1']", out)

    def test_peak_disk_and_tags(self):
        """Disk peak as final_MTfull; inference reads 50 member tags (11 MT F1 + 11 li2ft F2 + 28 old), every one a tag
        of final_MTfull or final_A22li_dec (so no new weight row), the same decoders and MAST net."""
        T = _module(self.pkg / "train.py", "pkg_train_mtli_test")
        self.assertEqual(T.peak_disk_gb(MTLI), T.PEAK_DISK_GB + 1)
        self.assertEqual(T.aux_groups(MTLI), ["mt22_F1", "mtftA22_F1"])
        I = _module(self.pkg / "infer.py", "pkg_infer_mtli_test")
        tags, full, dec = (I.recipe_tags(self.cfg(c)) for c in (MTLI, "final_MTfull", "final_A22li_dec"))
        self.assertEqual(len(tags), 50)
        self.assertEqual(len([t for t in tags if t.startswith("d3d_li2ftA22mt_shape")]), 11)
        self.assertEqual(len([t for t in tags if t.startswith("d3d_li2ftA22_notsL")]), 11)
        self.assertEqual(tags, {t for t in full if t.startswith("d3d_li2ftA22mt_shape")}
                         | {t for t in dec if not t.startswith("d3d_li2ftA22_shape")})
        self.assertTrue(tags <= full | dec)
        self.assertEqual(I.dec_tags(self.cfg(MTLI)), I.dec_tags(self.cfg("final_A22li_dec")))
        self.assertEqual(I.mast_tags(self.cfg(MTLI)), I.mast_tags(self.cfg("final_MTfull")))

    def test_weights_manifest_rows(self):
        """Inside a built package: every member / fallback tag final_MTli reads has a d3d row with a sha256."""
        f = REPO / "weights_manifest.json"
        if not (REPO / "configs" / f"{MTLI}.json").exists() or not f.exists():
            self.skipTest("weights_manifest.json is written by the package build")
        rows = json.loads(f.read_text())["files"]
        I = _module(self.pkg / "infer.py", "pkg_infer_mtli_rows_test")
        have = {r["tag"] for r in rows if r.get("group") == "d3d" and len(r.get("sha256") or "") == 64}
        self.assertEqual(I.recipe_tags(self.cfg(MTLI)) - have, set())


@unittest.skipUnless((REPO / "fec" / "final_pkg.py").exists(), "builder tests run in the research repo only")
class TestBuilder(unittest.TestCase):
    def test_weight_tags_drop_only_the_mt_bases(self):
        from fec import final_pkg as FP
        missing = [str(src) for src, *_ in FP.TRAIN_GROUPS.values() if not Path(src).exists()]
        if missing:
            self.skipTest(f"source job files absent: {missing[:2]}")
        with tempfile.TemporaryDirectory() as d:
            groups = FP.train_configs(Path(d))
        tags = FP.weight_tags(groups)
        every = {t for g in groups.values() for t in g["tags"]}
        mtb = {t for t in every if t.startswith("d3d_bxw3emt_")}
        self.assertEqual(len(mtb), 22)
        self.assertEqual(set(tags), every - mtb)
        self.assertEqual(len([t for t in tags if t.startswith("d3d_li2ftA22mt_")]), 22)
        without = {k: v for k, v in groups.items() if k not in FP.AUX_GROUPS}
        self.assertEqual(FP.weight_tags(without), sorted({t for g in without.values() for t in g["tags"]}))
        self.assertEqual(FP.TRAIN_ONLY_GROUPS, {"mt22_F1", "mt22_F2"})
        self.assertIn("mtask_aux_all", FP.ENTRIES)
        mods = FP.closure(FP.package_entries(False))
        self.assertTrue({"mtask_aux_all", "mtask_aux", "train_d3d_bx"} <= set(mods))
        self.assertIn("recipes_final/final_ip_C2_w3k22liAmtfull.json", FP.FEC_DATA)
        self.assertEqual(FP.CAND_GROUPS["final_MTfull"], MT_GROUPS)
        self.assertIn("final_A24r2", FP.package_cands(False))      # amendment 46: S2 = final_A24 + r = 2 maps
        self.assertEqual(FP.CAND_GROUPS["final_A24r2"], FP.CAND_GROUPS["final_A24"])
        import fec.run_final as RF
        a, b = (json.loads(json.dumps(RF.RECIPES[n])) for n in ("final_A24r2", "final_A24"))
        self.assertEqual(a["d3d"], b["d3d"])
        self.assertEqual(a["mast"]["composite"], RF.COMPOSITE_R2)
        self.assertEqual(RF.RECIPES["final_A24"]["mast"]["composite"], RF.COMPOSITE_V7H055)   # final_A24 unchanged

    def test_mtli_builder_constants(self):
        """final_MTli (amendment 50 (A)): shipped, its groups, its recipe in FEC_DATA, li2ftA22_F2 = the no-Thomson half
        of li2ftA22 (li2 cache via MT_LI2_GROUPS, LI2_CACHE_GROUPS unchanged), no new weight tag, the run_final preset =
        final_MTfull's chain with the dec block and the r = 2 composite."""
        from fec import final_pkg as FP
        self.assertIn(MTLI, FP.package_cands(False))
        self.assertEqual(FP.CAND_GROUPS[MTLI], MTLI_GROUPS)
        self.assertIn("recipes_final/final_ip_C2_w3k22liAmt.json", FP.FEC_DATA)
        src, pat, rep, gs = FP.TRAIN_GROUPS["li2ftA22_F2"]
        self.assertEqual((src, rep, gs), (FP.TRAIN_GROUPS["li2ftA22"][0], [], 4))
        self.assertEqual(pat, r"^d3d_li2ftA22_notsL_")
        self.assertEqual(FP.LI2_CACHE_GROUPS, {"li2ftA22"})
        self.assertIn("li2ftA22_F2", FP.MT_LI2_GROUPS)
        self.assertNotIn("li2ftA22_F2", FP.AUX_GROUPS | FP.TRAIN_ONLY_GROUPS)
        missing = [str(s) for s, *_ in FP.TRAIN_GROUPS.values() if not Path(s).exists()]
        if not missing:
            others = [c for c in FP.package_cands(False) if c != MTLI]
            with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
                every, without = FP.train_configs(Path(d1)), FP.train_configs(Path(d2), others)
            self.assertEqual(FP.weight_tags(every), FP.weight_tags(without))     # no new weights_manifest row
        import fec.run_final as RF
        mt, full = (json.loads(json.dumps(RF.RECIPES[n])) for n in (MTLI, "final_MTfull"))
        self.assertEqual(mt["d3d"]["dec"], RF.DEC_H1DEC)
        self.assertEqual(mt["mast"]["composite"], RF.COMPOSITE_R2)
        self.assertEqual(mt["d3d"]["b"], dict(full["d3d"]["b"], name="d3d_ip_C2_w3k22liAmt",
                                              recipe="fec/recipes_final/final_ip_C2_w3k22liAmt.json"))
        mt["d3d"].pop("b"), full["d3d"].pop("b")
        self.assertEqual(mt["d3d"], full["d3d"])
        self.assertEqual(mt["reference"], full["reference"])

    def test_source_job_files_are_the_production_lines(self):
        """The packaged MT lines come from the committed production job files (mtprod / mtf2 pods)."""
        from fec import final_pkg as FP
        for g, (src, pat, rep, gs) in FP.TRAIN_GROUPS.items():
            if g in FP.AUX_GROUPS:
                self.assertEqual(Path(src).parent.name, "mtprod" if g.endswith("F1") else "mtf2")
                self.assertEqual((rep, gs), ([], 4))


class TestAuxAll(unittest.TestCase):
    def _mod(self):
        try:
            from fec import mtask_aux_all as AA
        except Exception as e:  # pragma: no cover
            self.skipTest(f"fec.mtask_aux_all not importable: {e}")
        return AA

    def test_rows_land_at_their_cache_positions(self):
        """shot_jobs / compute: every li2-cache row gets the row the worker computed for (its shot, its frame)."""
        AA = self._mod()
        rng = np.random.default_rng(0)
        shot = np.repeat(np.arange(5), [3, 1, 4, 2, 5])
        perm = rng.permutation(len(shot))                     # ds order != shot-sorted order
        ds = {"shot": shot[perm], "fidx": np.arange(len(shot)) * 10 + 7,
              "info": {"order": [f"stem{i}" for i in range(5)]}}
        gl = np.arange(len(shot))

        def fake(args):
            key, fidx, pos, li2, src, R, Z = args
            return key, np.stack([np.full(20, 1000 * int(key[4:]) + f, float) for f in fidx])
        jobs, posd = AA.shot_jobs(ds, gl, "li2", "src", None, None)
        self.assertEqual(len(jobs), 5)
        with mock.patch.object(AA.MA, "_worker", fake):
            A = AA.compute(jobs, posd, len(gl), nproc=1, log_every=10 ** 9)
        want = 1000 * ds["shot"] + ds["fidx"]
        self.assertTrue(np.array_equal(A, np.repeat(want[:, None].astype(float), 20, 1)))
        sub = np.array([1, 4, 9])                             # a subset of positions
        jobs, posd = AA.shot_jobs(ds, gl, "li2", "src", None, None, pos=sub)
        with mock.patch.object(AA.MA, "_worker", fake):
            A = AA.compute(jobs, posd, len(gl), nproc=1, log_every=10 ** 9)
        self.assertTrue(np.array_equal(np.isfinite(A[:, 0]), np.isin(gl, sub)))
        self.assertTrue(np.array_equal(A[sub, 0], want[sub]))

    def test_write_cache_layout(self):
        AA = self._mod()
        with tempfile.TemporaryDirectory() as d:
            A = np.arange(40, dtype=float).reshape(2, 20)
            A[1, 3] = np.nan
            m = AA.write_cache(d, np.arange(2), A, np.array(["train", "test"]), {"x": 1})
            self.assertEqual(np.load(Path(d) / "gidx.npy").dtype, np.int64)
            self.assertEqual(np.load(Path(d) / "aux.npy").dtype, np.float32)
            self.assertEqual(m["stats"]["test"]["n_nan"][3], 1)
            self.assertEqual(json.loads((Path(d) / "meta.json").read_text())["aux_md5"],
                             hashlib.md5((Path(d) / "aux.npy").read_bytes()).hexdigest())

    def test_refuses_a_train_only_li2_cache(self):
        AA = self._mod()
        src = Path(AA.__file__).read_text(encoding="utf-8")
        self.assertIn('sorted(lm["splits"]) != ["test", "train", "val"]', src)
        self.assertIn("np.arange(len(ds[\"shot\"]))", src)


# ------------------------------------------------------------------------------- research repo, local artefacts
def _local_work():
    """The research machine's work root (fec.paths.WORK), None inside the package."""
    if not (REPO / "fec" / "final_pkg.py").exists():
        return None
    from fec.paths import WORK
    return Path(WORK) if Path(WORK).exists() else None


LOCAL = _local_work()


@unittest.skipUnless(LOCAL is not None, "research machine only")
class TestLocalArtefacts(unittest.TestCase):
    """Stored research-machine artefacts.  The three package-path runs below (2026-10-15/16) were made with the
    Development-phase MAST net (529ebce2, weights revision 95f88c25); their MAST assertions check that net's file
    12e264e5, not the net shipped at weights revision dfa0ee10 (README "Expected rebuild variance")."""

    def test_all_row_aux_cache_equals_production(self):
        new, prod = LOCAL / "mtpkg" / "aux_all_check", LOCAL / "mtprod" / "aux_all"
        if not (new / "meta.json").exists() or not (prod / "meta.json").exists():
            self.skipTest("local aux caches absent")
        a, b = json.loads((new / "meta.json").read_text()), json.loads((prod / "meta.json").read_text())
        self.assertEqual((a["aux_md5"][:8], a["gidx_md5"][:8]), PROD_AUX_MD5)
        self.assertEqual((a["aux_md5"], a["gidx_md5"]), (b["aux_md5"], b["gidx_md5"]))
        self.assertEqual(a["n"], 1559340)

    def test_stored_eqv_run_with_dev_phase_net_reproduced_the_dev_files(self):
        """Stored local run (2026-10-15, fec_work/mtpkg/eqv) of the package inference path (fec/run_final.py stages
        of final_MTfull on the 22 + 28 installed members and the shipped decoders, with the Development-phase MAST net
        of weights revision 95f88c25): DIII-D = the Dev-read MT-full file, MAST = the r = 2 Dev file 12e264e5, no
        fallback.  Not a check of the MAST net shipped at dfa0ee10 (README "Expected rebuild variance")."""
        root = LOCAL / "mtpkg" / "eqv"
        sub = root / "submission_mtfull"
        if not (sub / "diii_d_public_test.npz").exists():
            self.skipTest("equivalence run absent")

        def sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            return h.hexdigest()
        self.assertEqual(sha(sub / "diii_d_public_test.npz"), DEV_D3D_SHA)
        self.assertEqual(sha(sub / "mast_public_test.npz"), DEV_MAST_SHA)
        dec = json.loads((root / "manifest" / "d3d_dec.json").read_text())
        self.assertEqual(dec["status"], "ok")
        self.assertIsNone(dec["info"].get("fallback"))
        meta = json.loads(next((root / "preds").glob("*_dec_lirep4/diii_d_public_test_lirep4_meta.json")).read_text())
        self.assertEqual((meta["triggered"], meta["repaired"], meta["left_out"]), (9, 9, 0))

    def test_stored_a24r2_run_with_dev_phase_net_reproduced_dev_mast_file(self):
        """Stored local run (2026-10-15, fec_work/mtpkg/eqv_a24r2) of the package inference path for final_A24r2
        (mast_amort -> mast_composite -> submission on the installed models, with the Development-phase MAST net of
        weights revision 95f88c25; DIII-D half = final_A24's shipped-weights preds): MAST npz = the r = 2 Dev file
        12e264e5.  Not a check of the MAST net shipped at dfa0ee10 (README "Expected rebuild variance")."""
        sub = LOCAL / "mtpkg" / "eqv_a24r2" / "submission_a24r2"
        if not (sub / "mast_public_test.npz").exists():
            self.skipTest("final_A24r2 byte-check run absent")
        h = hashlib.sha256()
        with open(sub / "mast_public_test.npz", "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        self.assertEqual(h.hexdigest(), DEV_MAST_SHA)

    def test_stored_mtli_run_with_dev_phase_net_reproduced_the_dev_files(self):
        """Stored local run (2026-10-16, fec_work/mtlipkg/eqv_mtli) of the final_MTli package inference path on the
        installed members (d3d_b -> q95h -> d3d_dec with the shipped decoders -> lirep4 -> MAST with the
        Development-phase MAST net of weights revision 95f88c25 -> submission; agent mtlipkg): DIII-D = 728f71e1,
        MAST = 12e264e5.  Not a check of the MAST net shipped at dfa0ee10 (README "Expected rebuild variance")."""
        root = LOCAL / "mtlipkg" / "eqv_mtli"
        sub = root / "submission_mtli"
        if not (sub / "diii_d_public_test.npz").exists() or not (sub / "mast_public_test.npz").exists():
            self.skipTest("final_MTli byte-check run absent")

        def sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            return h.hexdigest()
        self.assertEqual(sha(sub / "diii_d_public_test.npz"), MTLI_D3D_SHA)
        self.assertEqual(sha(sub / "mast_public_test.npz"), DEV_MAST_SHA)
        dec = json.loads((root / "manifest" / "d3d_dec.json").read_text())
        self.assertEqual(dec["status"], "ok")
        self.assertIsNone(dec["info"].get("fallback"))
        meta = json.loads(next((root / "preds").glob("*_dec_lirep4/diii_d_public_test_lirep4_meta.json")).read_text())
        self.assertEqual((meta["triggered"], meta["repaired"], meta["left_out"]), (10, 10, 0))


if __name__ == "__main__":
    unittest.main()
