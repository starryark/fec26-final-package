# `fusion_scoring/` — the competition metric, vendored

These files are **copied unmodified** from the scoring program Codabench runs. They are what
`local_score.py` uses, so the functionals you score against locally are the same code that scores
your submission — not a reimplementation that can drift.

```
common.py     weights, scalar names, per-machine sign conventions
contour.py    marching squares + symmetric Hausdorff distance
o_point.py    magnetic-axis (O-point) finder
lcfs.py       last-closed-flux-surface extraction
derive.py     the seven psi-derived scalars: R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li
metrics.py    pooled R2 accumulation and the composite S
masks/        per-machine plasma-envelope mask (65x65 bool) used by the LCFS extractor
```

Only **numpy** is required — `contour.py` implements marching squares itself, so there is no
scipy or scikit-image dependency.

## What is NOT here

The ground truth. The platform's `io_ref.py` reads reference bundles holding the withheld test
targets; nothing in this directory can reach them. `local_score.py` builds its targets from the
**training** shots you point it at, by running these same functions over ground truth you already
have.

That is also why a local score is a proxy, not a leaderboard prediction: identical code, different
data.

## The envelope mask

`masks/*_envelope.npz` is the union of every ground-truth LCFS polygon across the machine's corpus,
rasterized to the 65×65 grid and dilated two cells. It caps how far the extractor's boundary
bisection can grow so the surface tracks the plasma instead of ballooning to the grid wall. Being a
corpus-wide union rather than per-shot, it carries no per-shot information — it is the same mask
the platform uses.

## Keeping it in sync (organizers)

If the scoring program changes, re-copy — do not hand-edit these files:

```bash
cp codabench/bundle/scoring_program/{common,contour,o_point,lcfs,derive,metrics}.py \
   fusion-equilibrium-challenge-starter/fusion_scoring/
```

Then re-run the harness self-checks, which assert the same invariants as the organizers'
`ref_builder/qa_selfscore.py`:

```bash
python local_score.py --mode perfect --n-shots 2      # must print S = 1.0
python local_score.py --mode zeros   --n-shots 2      # must print S = 0.0
```

A perfect prediction scores exactly 1 only because both sides go through identical code. If that
check stops passing after a sync, the vendored copy and the platform have diverged.
