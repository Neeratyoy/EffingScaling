"""Search for a functional form for the compute -> ROC AUC Pareto frontier.

Data: `output/nanotabpfn/real_data_roc_auc_big_prior_pareto.csv` (written by
`nanotabpfn_analysis.ipynb`) -- 15 Pareto points, FLOPs 1e11..1e15, ROC AUC 0.54..0.86.

Every candidate form carries an **irreducible term**, reported as a ceiling on ROC AUC
(`ceiling = 1 - E`). Forms without one are kept only as explicitly-labelled references.
The two-stage forms fit the prior validation loss first; that stage carries its own
irreducible loss `E_L`.

Three things are measured for each form, in decreasing order of how much they should be
trusted:

1. **holdout MAE** -- refit on the low-compute points only and predict the rest, at two
   cuts (<= 3.2e13, leaving 6 test points, and <= 1e14, leaving 4); the headline is their
   mean. This is the only number that tests extrapolation, which is what a form is for.
2. **ceiling profile** -- refit with the ceiling clamped across a grid, and report the
   range of ceilings whose loss stays within 10% of the best. A ceiling is pinned down
   by data past the point where the curve flattens; this frontier has not flattened, so
   the range matters more than the point estimate.
3. **in-sample R^2 / RMSE** -- cheap to inflate with parameters, listed last on purpose.

Run:  python sigmoid_fit.py [--round N]

Figures are written per round to `figures/nanotabpfn/functional_forms/round_<N>/`, so
earlier rounds are not overwritten.

SEARCH LOG (outcomes, not plans)
--------------------------------
Round 1 -- published baselines. Every single-transition form leaves the same residual
    signature: it cannot be both flat near chance below 1e12.5 and shallow above 1e13.
    The best of them (BNSL with one break) reaches holdout MAE 0.009 but pins its ceiling
    on the lower bound, i.e. it saturates exactly at the last observed point. Gadre's
    exponential link fails outright (MAE 0.150) -- the validation loss on this frontier
    spans 4.8 -> 0.6, and an exponential in the loss cannot bend twice.
Round 2 -- more than one transition. BNSL with two breaks wins by a factor of ~1.5 on
    holdout (0.0061) and is the ONLY form in the whole search whose ceiling profile has
    walls on both sides: an actually identified irreducible loss at 0.887 ROC AUC,
    tolerated range [0.880, 0.901]. Multi-step variants fit in-sample (R^2 0.998) but
    extrapolate worse, and their ceilings run away to 1.
Round 3 -- reductions. All four reduced two-break variants keep the ceiling (0.884-0.887,
    still identified), so the ceiling is a property of the data rather than of the
    9-parameter freedom. But every reduction costs extrapolation accuracy (holdout MAE
    0.032-0.036 vs 0.0061), and `flat + shared f` is numerically unstable -- its refit on
    8 points diverges. Reductions were not adopted.
Round 4 -- ceiling treated as known (held at 0.887) to buy back a parameter. It does not
    pay: every anchored form lands at holdout MAE 0.020-0.038, worse than the free
    two-break fit. So the two-break *shape* is what earns the accuracy, not the ceiling.

CONCLUSION (rounds 1-4): BNSL with two breaks (Caballero et al. at n = 2), fitted on the
gap 1 - ROC AUC with an irreducible term. Irreducible ROC AUC = 0.887, tolerated [0.880,
0.901]. That ceiling coincides with TabPFN's own TabArena score (0.8874) to three decimals
-- which is either a meaningful statement about the prior or a coincidence on 15 points,
and this data cannot tell you which.

Round 5 (2026-10-06) -- re-examination. The rounds-1-4 result does not survive. The
    big-prior subset (16 features, 400 rows) drops the whole 1e13 stage (s1 has at most 8
    features), so below 1e13.5 FLOPs it trails the all-runs frontier by up to 0.20 ROC AUC;
    its low-compute plateau is a selection artefact, and 8 of its 20 points come from one
    run. `--frontier all` uses every run (22 points from 18 runs, <= 4 checkpoints each).
    Round 5 adds bounded forms from the learning-curve and ICL-scaling literature, a
    logit-space power law, two-stage forms through the real-data NLL and a loss-to-loss
    chain (prior CE -> real NLL -> AUC); staged forms now refit every stage inside each
    holdout. It also adds a leave-future-out ladder, leave-one-run-out, a bootstrap over
    runs, a top-k frontier (`--topk`) and seed-noise weights (`--seed-weights`).
    Two leaks were then found and fixed. (1) Ceiling and floor bounds were built from the
    best ROC AUC of the whole frontier, so every holdout refit had to put its ceiling above
    the best *test* point; 13 of 25 free-ceiling forms sat exactly on that bound at the
    3.2e13 cut, and rounds 1-4 inherited this. `bounds_for` now re-ties them to the training
    points. (2) A first version of the ladder seeded refits with the full-data optimum;
    over-parameterised forms barely move from it. `_refit_starts` now uses the grid only.
    Ranked by the leave-future-out ladder (long horizon = fit up to 1e13 FLOPs), the logit
    power law (3p) is the best free-ceiling form on the all-runs frontier (0.024 MAE), the
    top-5-mean frontier (0.039) and with seed-noise weights (0.025). Two-stage forms through
    the real NLL tie it on one frontier only; the loss-to-loss chain adds nothing (the real
    NLL floor goes to 0); BNSL-2 diverges at long horizons everywhere. Ceilings of the
    better forms: 0.88-0.91, and refits on growing prefixes show the ceiling tracking the
    best observed point upward instead of converging. Write-up:
    output/nanotabpfn/paper/functional_forms_{subsection,appendix}.tex.

References
----------
Kaplan et al. 2020, arXiv:2001.08361          bare power law
Hoffmann et al. 2022, arXiv:2203.15556        power law + irreducible term
Alabdulmohsin et al. 2022, arXiv:2209.06640   M4: (y - E) / (I - y)^a = b x^c
Caballero et al. 2022, arXiv:2210.14891       BNSL: smoothly broken power law
Michaud et al. 2023, arXiv:2303.13506         quantization model: loss as a sum of
                                              discrete quanta -> multi-break forms
Owen 2024, arXiv:2401.04757                   sigmoid in log-compute for benchmarks
Gadre et al. 2024, arXiv:2403.08540           downstream error = exp(-k L^g + b)
Ruan et al. 2024, arXiv:2405.10938            observational laws, sigmoid link
Rosenfeld et al. 2020, arXiv:1909.12673       envelope eps0 |e| / |e - i eta|: bounded by chance
Arora et al. 2024, arXiv:2410.16531           bounded power law K + c (1 + x/x0)^-a (ICL)
Viering & Loog 2021, arXiv:2103.10948         learning-curve catalogue: POW4, WBL4, MMF4
Brandfonbrener et al. 2024, arXiv:2411.12925  loss-to-loss: L1 = K (L0 - E0)^kappa + E1
arXiv:2605.09189                              bounded "practical" law E + (L0 - E) h / (1 + h)
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import cycle, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from sklearn.metrics import r2_score

from scaling.utils import fit_parametric_form, functional_form_L0, sigmoid_functional_form

# --- paths -------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "output" / "nanotabpfn" / "real_data_roc_auc_big_prior_pareto.csv"
TABPFN_BASELINE_PATH = HERE.parent / "data" / "summary_tabpfn.csv"
FIGURE_ROOT = HERE / "figures" / "nanotabpfn" / "functional_forms"

ALL_RUNS_PATH = HERE.parent / "data" / "nanotabpfn_data_preprocessed.csv"

FLOPS_COLUMN = "total_flops_rounded"
LOSS_COLUMN = "real_data/roc_auc"
VAL_LOSS_COLUMN = "val/val_loss"
NLL_COLUMN = "real_data/nll"
# Reference compute for the centred forms: ln(C / C_REF) keeps amplitude and exponent from
# being collinear, which is what pins the L0 amplitude to its start value when ln C ~ 30.
C_REF = 1e13

# Huber delta used throughout the notebook fits; kept identical for comparability.
HUBER_DELTA = 1e-3
EXTRAPOLATION_FLOPS = 1e19
# Points above each threshold are held out of the fit and used to score extrapolation.
# Two cuts: 1e14 leaves 4 test points, 3.2e13 leaves 6 and is the harder test.
HOLDOUT_FLOPS_LIST = [3.16e13, 1e14]
HOLDOUT_FLOPS = max(HOLDOUT_FLOPS_LIST)
# ROC AUC of a random classifier: the floor any of these curves starts from.
CHANCE_AUC = 0.5
# The ceiling every two-break variant in rounds 2 and 3 converged on (0.884-0.887).
# Round 4 treats it as known rather than fitted.
ANCHOR_CEILING = 0.887

# Ceiling profiling.
CEILING_GRID_SIZE = 21
CEILING_TOLERANCE = 1.10
MAX_CEILING = 0.9999
MIN_LOG_GAP = np.log(1e-6)
# Cap on optimizer restarts per fit; larger grids are thinned deterministically.
MAX_STARTS = 500
PROFILE_SEEDS = 8
# Minimum training points a holdout refit needs to be meaningful.
MIN_HOLDOUT_TRAIN = 4
# Leave-future-out ladder: every cut that leaves at least this many test points.
MIN_LADDER_TEST = 3
# Refits inside the ladder / leave-one-run-out start from the initial grid thinned to this many
# starts; the bootstrap (not a holdout) starts from the full fit plus BOOTSTRAP_SEEDS grid points.
REFIT_SEEDS = 100
BOOTSTRAP_SEEDS = 10
# Top-k frontier: a FLOP level needs at least this many runs per averaged run, or it is dropped
# (the extreme-value bias of a max over K runs, and the mean of the top k, both need K >> k).
MIN_RUNS_PER_TOPK = 4
BOOTSTRAP_SEED = 0
# Fitted floors (starting ROC AUC) must lie this far below the best observed point.
FLOOR_MARGIN = 0.05
# Seed noise of the frontier ROC AUC (coefficient of variation per FLOP level) from the s1 seed
# study; levels beyond the table reuse its last value.
SEED_CV_PATH = HERE / "figures" / "nanotabpfn" / "seeds" / "seed_variance" / "relative_std.txt"
# Per-point noise in ROC AUC units, keyed by FLOP level; empty means unweighted fits.
NOISE: dict[str, dict] = {"sigma_auc": {}, "auc": {}}

LN10 = np.log(10.0)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DASHES = [
    "-",
    (0, (6, 2)),
    (0, (1, 1.6)),
    (0, (4, 1.5, 1, 1.5)),
    (0, (7, 1.5, 1, 1.5)),
    (0, (3, 1.2)),
    (0, (2, 1, 5, 1)),
    (0, (8, 2, 2, 2)),
]

ROUND_2_NOTES = """\
Round 1 said: every one-transition form leaves the same residual signature -- it cannot
be both flat near chance below 1e12.5 and shallow above 1e13. So round 2 adds forms with
more than one transition (BNSL with two breaks, and a two-step version of the step form,
both motivated by Michaud et al.'s quantization picture of scaling as a sum of discrete
capability jumps), plus M4 with its floor pinned at chance AUC = 0.5 rather than fitted.
"""

ROUND_3_NOTES = """\
Round 2's two-break BNSL was the only form with an interior minimum in its ceiling
profile -- i.e. the only one whose irreducible loss the data actually pins down, at a
ceiling near 0.887 ROC AUC. It spends 9 parameters on 15 points, so round 3 asks whether
that identification survives when parameters are taken away: one sharpness instead of
two, a flat start (c0 = 0) instead of a decaying first regime, both at once, and finally
the sharpness fixed outright. If the ceiling stays put and stays identified, it is a
property of the data; if it moves or the walls vanish, it was a property of the freedom.
"""


ROUND_4_NOTES = """\
Rounds 2 and 3 agreed on the ceiling -- every two-break variant landed at 0.884-0.887 ROC
AUC with walls on both sides of its profile -- but the parameter reductions that made the
form cheaper all made its extrapolation worse. Round 4 tries the other trade: hold the
ceiling at the identified value and give the freed parameter back to the shape. If a
four-parameter anchored form extrapolates as well as the nine-parameter free one, the
ceiling was doing the work all along, and that is the form to use.
"""


# --- functional forms ---------------------------------------------------------
# Gap forms are fitted on y = log(1 - ROC AUC) and return that log-gap when
# return_log_loss=True, matching scaling.utils.functional_form_L0. An irreducible gap
# enters as logaddexp(log_e, <decaying part>), the stable way to write E + decaying.
def linear_in_log_flops(data, params, return_log_loss: bool = True):
    """Trivial baseline: ROC AUC linear in log FLOPs, unbounded above.

    AUC = a + b * log C. No ceiling, no floor -- included only to show what the more
    structured forms have to beat. Params: [a, b].
    """
    a, b = params
    return a + b * np.log(data)


def powerlaw_gap(data, params, return_log_loss: bool = True):
    """Bare power law on the gap (Kaplan et al. 2020 applied to 1 - ROC AUC).

    1 - AUC = exp(a) * C^b   <=>   log(1 - AUC) = a + b * log C

    Reference for a frontier with *no* irreducible loss: ROC AUC -> 1. Params: [a, b].
    """
    a, b = params
    log_gap = a + b * np.log(data)
    return log_gap if return_log_loss else np.exp(log_gap)


def m4_gap(data, params, return_log_loss: bool = True):
    """M4 (Alabdulmohsin et al. 2022) on the gap g = 1 - ROC AUC.

    (g - E) / (I - g)^alpha = exp(log_b) * C^c

    E is the irreducible gap and I the gap of a random classifier (= 1 - CHANCE_AUC), so
    the curve runs from chance at small C to the ceiling 1 - E at large C. At alpha = 1
    this solves in closed form to the four-parameter sigmoid; at other alpha the approach
    to the ceiling is faster or slower than logistic, which is the whole point of the
    form. Solved for g by bisection, since no closed form exists for general alpha.
    Params: [log_e, log_b, c, alpha].
    """
    log_e, log_b, c, alpha = params
    e_gap = np.exp(log_e)
    i_gap = 1.0 - CHANCE_AUC
    rhs = np.exp(log_b + c * np.log(data))

    # F(g) = (g - E) - rhs * (I - g)^alpha is increasing in g on (E, I): bisect for its root.
    lo = np.full_like(rhs, e_gap + 1e-12)
    hi = np.full_like(rhs, i_gap - 1e-12)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = (mid - e_gap) - rhs * np.power(np.maximum(i_gap - mid, 1e-300), alpha)
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    gap = 0.5 * (lo + hi)
    return np.log(gap) if return_log_loss else gap


def bnsl_gap(data, params, return_log_loss: bool = True):
    """BNSL (Caballero et al. 2022) on the gap, one break, with an irreducible term.

    1 - AUC = E + exp(log_b) * C^-c0 * (1 + (C/d)^(1/f))^(-c1 * f)

    The exponent moves smoothly from -c0 below the break at d to -(c0 + c1) above it, f
    setting the sharpness. Params: [log_e, log_b, c0, c1, log_d, log_f].
    """
    log_e, log_b, c0, c1, log_d, log_f = params
    log_c = np.log(data)
    softplus = np.logaddexp(0.0, (log_c - log_d) / np.exp(log_f))
    log_gap = np.logaddexp(log_e, log_b - c0 * log_c - c1 * np.exp(log_f) * softplus)
    return log_gap if return_log_loss else np.exp(log_gap)


def bnsl_gap_2break(data, params, return_log_loss: bool = True):
    """BNSL with two breaks -- the published product form at n = 2.

    1 - AUC = E + exp(log_b) * C^-c0 * prod_i (1 + (C/d_i)^(1/f_i))^(-c_i * f_i)

    Two breaks let the exponent go shallow -> steep -> shallow, which is the shape this
    frontier actually shows and which one break cannot produce.
    Params: [log_e, log_b, c0, c1, log_d1, log_f1, c2, log_d2, log_f2].
    """
    log_e, log_b, c0, c1, log_d1, log_f1, c2, log_d2, log_f2 = params
    log_c = np.log(data)
    f1, f2 = np.exp(log_f1), np.exp(log_f2)
    term = log_b - c0 * log_c
    term = term - c1 * f1 * np.logaddexp(0.0, (log_c - log_d1) / f1)
    term = term - c2 * f2 * np.logaddexp(0.0, (log_c - log_d2) / f2)
    log_gap = np.logaddexp(log_e, term)
    return log_gap if return_log_loss else np.exp(log_gap)


def make_bnsl_2break(flat_start: bool = False, shared_f: bool = False, fixed_log_f: float | None = None):
    """Reduced parameterizations of the two-break BNSL.

    Round 2's two-break BNSL was the only form whose ceiling profile had walls on both
    sides -- an actually identified irreducible loss -- but it spends 9 parameters on 15
    points, so that identification could be an artefact of the freedom rather than a fact
    about the data. These variants remove parameters one at a time to see if it survives:

    flat_start   c0 = 0, so the gap is constant below the first break. That is what the
                 low-compute plateau looks like, rather than a slowly decaying power law.
    shared_f     one sharpness for both breaks instead of two.
    fixed_log_f  sharpness not fitted at all: both breaks are sharp.

    Returns (form, param_names); the names drive the bounds and grid lookups below.
    """
    names = ["log_e", "log_b"]
    if not flat_start:
        names.append("c0")
    names += ["c1", "log_d1", "c2", "log_d2"]
    if fixed_log_f is None:
        names += ["log_f"] if shared_f else ["log_f1", "log_f2"]

    def form(data, params, return_log_loss: bool = True):
        p = dict(zip(names, params))
        log_c = np.log(data)
        c0 = 0.0 if flat_start else p["c0"]
        if fixed_log_f is not None:
            f1 = f2 = np.exp(fixed_log_f)
        elif shared_f:
            f1 = f2 = np.exp(p["log_f"])
        else:
            f1, f2 = np.exp(p["log_f1"]), np.exp(p["log_f2"])
        term = p["log_b"] - c0 * log_c
        term = term - p["c1"] * f1 * np.logaddexp(0.0, (log_c - p["log_d1"]) / f1)
        term = term - p["c2"] * f2 * np.logaddexp(0.0, (log_c - p["log_d2"]) / f2)
        log_gap = np.logaddexp(p["log_e"], term)
        return log_gap if return_log_loss else np.exp(log_gap)

    return form, names


def _d_anchored(p):
    return "ceiling held fixed; " + ", ".join(f"{v:.4g}" for v in p)


def _d_bnsl2_named(names):
    """Describe a reduced two-break BNSL by name, since its parameters vary by variant."""

    def describe(p):
        d = dict(zip(names, p))
        c0 = d.get("c0", 0.0)
        return (
            f"E={np.exp(d['log_e']):.4g}, exponent -{c0:.4f} -> -{c0 + d['c1']:.4f} -> "
            f"-{c0 + d['c1'] + d['c2']:.4f}, breaks 1e{d['log_d1'] / LN10:.2f} and "
            f"1e{d['log_d2'] / LN10:.2f}"
        )

    return describe


def stepped_powerlaw_gap(data, params, return_log_loss: bool = True):
    """Power-law gap with one *transient* step, plus an irreducible term.

    1 - AUC = E + exp(a + b*logC + d*sigmoid(-k*(logC - log_x0)))

    Not a published form: it is BNSL with the exponent change made temporary, so the
    frontier is a power law of slope b interrupted by a single drop of size d in the
    log-gap. Params: [log_e, a, b, d, log_x0, k].
    """
    log_e, a, b, d, log_x0, k = params
    log_c = np.log(data)
    log_gap = np.logaddexp(log_e, a + b * log_c + d * expit(-k * (log_c - log_x0)))
    return log_gap if return_log_loss else np.exp(log_gap)


def two_stepped_powerlaw_gap(data, params, return_log_loss: bool = True):
    """The step form with two steps -- one capability jump per transition.

    1 - AUC = E + exp(a + b*logC + d1*sigmoid(-k1*(logC - x1)) + d2*sigmoid(-k2*(logC - x2)))

    Motivated by Michaud et al.'s quantization picture, where the scaling curve is a sum
    of discrete jumps rather than one smooth bend.
    Params: [log_e, a, b, d1, log_x1, k1, d2, log_x2, k2].
    """
    log_e, a, b, d1, log_x1, k1, d2, log_x2, k2 = params
    log_c = np.log(data)
    term = a + b * log_c + d1 * expit(-k1 * (log_c - log_x1)) + d2 * expit(-k2 * (log_c - log_x2))
    log_gap = np.logaddexp(log_e, term)
    return log_gap if return_log_loss else np.exp(log_gap)


def make_two_stage_logistic(loss_law: Callable[[np.ndarray], np.ndarray]):
    """Two-stage, logistic link (Owen 2024 / Ruan et al. 2024).

    Stage 1 (fitted separately, on the prior validation loss):  L(C) = E_L + A * C^-alpha
    Stage 2:  AUC = a_min + (ceiling - a_min) * sigmoid(-k * (L(C) - L_mid))

    The benchmark metric is predicted from the model's *loss*, not from compute directly,
    which is how the LLM literature does it; the compute dependence is entirely in stage
    1, whose irreducible loss E_L is reported alongside the ceiling.
    Params: [ceiling, a_min, k, l_mid].
    """

    def form(data, params, return_log_loss: bool = True):
        ceiling, a_min, k, l_mid = params
        predicted_loss = loss_law(np.asarray(data, dtype=float))
        return a_min + (ceiling - a_min) * expit(-k * (predicted_loss - l_mid))

    return form


def make_two_stage_exponential(loss_law: Callable[[np.ndarray], np.ndarray]):
    """Two-stage, exponential link (Gadre et al. 2024, their Eq. 5).

    Stage 1:  L(C) = E_L + A * C^-alpha            (their Eq. 4 at fixed token multiplier)
    Stage 2:  Err(L) = eps - k * exp(-gamma * L)

    Their epsilon is the error at infinite loss and `eps - k` the error at zero loss, so
    the irreducible gap is E = eps - k. Rewriting around it,

        1 - AUC = E + k * (1 - exp(-gamma * L(C)))

    which puts the irreducible term first, as every other form here does.
    Params: [log_e, k, gamma].
    """

    def form(data, params, return_log_loss: bool = True):
        log_e, k, gamma = params
        predicted_loss = np.maximum(loss_law(np.asarray(data, dtype=float)), 1e-9)
        gap = np.exp(log_e) + k * (-np.expm1(-gamma * predicted_loss))
        gap = np.clip(gap, 1e-12, 1.0 - 1e-12)
        return np.log(gap) if return_log_loss else gap

    return form


def m4_gap_free_floor(data, params, return_log_loss: bool = True):
    """M4 with the floor fitted instead of pinned at chance.

    (g - E) / (I - g)^alpha = exp(log_b) * C^c,  I = 1 - floor_auc

    Round 1 showed the published M4, whose floor is the random-guess level, cannot sit on
    the low-compute points: those plateau at ROC AUC 0.54-0.59, visibly above chance, so
    the effective starting point is not 0.5. Params: [log_e, log_b, c, alpha, floor_auc].
    """
    log_e, log_b, c, alpha, floor_auc = params
    e_gap = np.exp(log_e)
    i_gap = 1.0 - floor_auc
    rhs = np.exp(log_b + c * np.log(data))

    lo = np.full_like(rhs, e_gap + 1e-12)
    hi = np.full_like(rhs, max(i_gap - 1e-12, e_gap + 2e-12))
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = (mid - e_gap) - rhs * np.power(np.maximum(i_gap - mid, 1e-300), alpha)
        lo = np.where(f < 0, mid, lo)
        hi = np.where(f < 0, hi, mid)
    gap = 0.5 * (lo + hi)
    return np.log(gap) if return_log_loss else gap


# --- round 5: bounded forms from the learning-curve / ICL literature ------------
def _lnc(data) -> np.ndarray:
    """Centred log compute, ln(C / C_REF)."""
    return np.log(np.asarray(data, dtype=float) / C_REF)


def bounded_powerlaw_gap(data, params, return_log_loss: bool = True):
    """Bounded power law on the gap (Arora et al. 2024; LCDB's POW4 in another parameterisation).

    1 - AUC = E + exp(log_c) * (1 + C/C0)^-alpha

    Finite at C -> 0 (gap E + c, i.e. a plateau rather than a divergence), power-law tail
    above C0. It is BNSL with one break, a flat start and the sharpness fixed at 1, so it is
    the 4-parameter member of that family. Params: [log_e, log_c, log_c0, alpha].
    """
    log_e, log_c, log_c0, alpha = params
    log_gap = np.logaddexp(log_e, log_c - alpha * np.logaddexp(0.0, np.log(data) - log_c0))
    return log_gap if return_log_loss else np.exp(log_gap)


def rosenfeld_envelope_gap(data, params, return_log_loss: bool = True):
    """Rosenfeld et al. 2020 envelope on the gap: a power law bounded by the chance level.

    gap = eps0 * e / sqrt(e^2 + eta^2),   e = E~ + a * (C/C_REF)^-b,   eps0 = 1 - CHANCE_AUC

    The modulus trick saturates smoothly at eps0 for small C and tracks the power law e for
    large C. The ceiling parameter is the irreducible gap g_inf = exp(log_e), from which the
    power law's own offset follows as E~ = g_inf * eta / sqrt(eps0^2 - g_inf^2).
    Params: [log_e, log_a, b, log_eta].
    """
    log_e, log_a, b, log_eta = params
    eps0 = 1.0 - CHANCE_AUC
    g_inf = min(np.exp(log_e), eps0 - 1e-9)
    eta = np.exp(log_eta)
    e_tilde = g_inf * eta / np.sqrt(eps0**2 - g_inf**2)
    e = e_tilde + np.exp(log_a - b * _lnc(data))
    gap = eps0 * e / np.sqrt(e**2 + eta**2)
    return np.log(gap) if return_log_loss else gap


def bounded_sigmoid_chance_gap(data, params, return_log_loss: bool = True):
    """The "practical" bounded law (arXiv:2605.09189) with the uninformed level at chance.

    gap = E + (eps0 - E) * h / (1 + h),   h = exp(log_a - b * ln(C/C_REF))

    Equivalently M4 at alpha = 1, or the logistic in log C with its floor pinned at chance:
    the minimal bounded form, 3 parameters. Params: [log_e, log_a, b].
    """
    log_e, log_a, b = params
    eps0 = 1.0 - CHANCE_AUC
    e = np.exp(log_e)
    gap = e + (eps0 - e) * expit(log_a - b * _lnc(data))
    return np.log(gap) if return_log_loss else gap


def make_weibull_gap(free_floor: bool):
    """WBL4 from the learning-curve catalogue (Viering & Loog 2021) on the gap.

    gap = E + (g0 - E) * exp(-a * (C/C_REF)^d)

    A stretched-exponential approach to the ceiling instead of a power-law one. g0 is the
    gap at C -> 0: chance (3 params) or fitted (4 params).
    Params: [log_e, log_a, d] or [log_e, log_a, d, floor_auc].
    """

    def form(data, params, return_log_loss: bool = True):
        if free_floor:
            log_e, log_a, d, floor_auc = params
            g0 = 1.0 - floor_auc
        else:
            log_e, log_a, d = params
            g0 = 1.0 - CHANCE_AUC
        e = np.exp(log_e)
        gap = e + (g0 - e) * np.exp(-np.exp(log_a + d * _lnc(data)))
        gap = np.maximum(gap, 1e-12)
        return np.log(gap) if return_log_loss else gap

    return form


def logit_powerlaw(data, params, return_log_loss: bool = True):
    """Power law with an irreducible term in logit space (the bounded analogue of the
    log-accuracy law of arXiv:2512.08894).

    logit(AUC) = l_inf - exp(log_a) * (C/C_REF)^-b

    Ceiling expit(l_inf); no chance floor (AUC -> 0 as C -> 0). Fitted in logit space.
    Params: [l_inf, log_a, b].
    """
    l_inf, log_a, b = params
    z = l_inf - np.exp(log_a - b * _lnc(data))
    return z if return_log_loss else expit(z)


def _logit(y):
    y = np.clip(np.asarray(y, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(y / (1 - y))


def centred_l0(data, params, return_log_loss: bool = True):
    """L = E + exp(a) * (C/C_REF)^-b, the L0 form with centred compute. Params: [l0, a, b]."""
    l0, a, b = params
    log_l = np.logaddexp(l0, a - b * _lnc(data))
    return log_l if return_log_loss else np.exp(log_l)


def loss_to_loss(prior_loss, params, return_log_loss: bool = True):
    """Brandfonbrener et al. 2024 shifted power law between two losses.

    L_real = E_real + K * (L_prior - E_prior)^kappa

    `prior_loss` is the column stack [L_prior, E_prior]. Params: [log_e_real, log_k, kappa].
    """
    log_e_real, log_k, kappa = params
    excess = np.maximum(prior_loss[:, 0] - prior_loss[:, 1], 1e-9)
    log_l = np.logaddexp(log_e_real, log_k + kappa * np.log(excess))
    return log_l if return_log_loss else np.exp(log_l)


L0_CENTRED_GRID = [
    [np.log(e) for e in (0.05, 0.2, 0.4, 0.7)],
    [np.log(a) for a in (0.1, 0.5, 1.0, 3.0)],
    [0.05, 0.1, 0.3, 0.6],
]
L2L_GRID = [[np.log(e) for e in (0.05, 0.15, 0.25)], [-1.0, 0.0, 1.0], [0.5, 1.0, 2.0]]


def fit_loss_law(flops: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, Callable]:
    """L(C) = E + A * (C/C_REF)^-alpha on a loss column, with E bounded below the smallest
    observed loss. Returns ([l0, a, b] in the centred parameterisation, callable L(C))."""
    params, _ = fit_parametric_form(
        func_form=centred_l0,
        X_data=flops,
        y_data=np.log(loss),
        initial_grid=list(product(*L0_CENTRED_GRID)),
        bounds=[(np.log(1e-4), np.log(loss.min())), (-20.0, 20.0), (1e-3, 5.0)],
        delta=HUBER_DELTA,
        use_scipy=True,
    )

    def law(c: np.ndarray) -> np.ndarray:
        return centred_l0(np.asarray(c, dtype=float), params, return_log_loss=False)

    return params, law


class StageContext:
    """Earlier stages of a staged form, refitted on whatever subset the final stage sees.

    `fit(mask)` refits the stage laws on the points selected by `mask` and stores a callable
    C -> predicted feature in `feature`; the final-stage form reads it at call time. Rounds
    1-4 fitted stage 1 once on all points, so the holdout scores of the two-stage forms were
    contaminated by the test points' validation loss. Round 5 refits inside each holdout.

    chain = "val": feature = prior validation loss law L_val(C)
    chain = "nll": feature = real-data NLL law L_nll(C)
    chain = "l2l": feature = loss-to-loss chain, L_nll = E_real + K (L_val(C) - E_val)^kappa
    """

    def __init__(self, flops: np.ndarray, val_loss: np.ndarray, nll: np.ndarray, chain: str):
        if chain not in ("val", "nll", "l2l"):
            raise ValueError(chain)
        self.flops, self.val_loss, self.nll, self.chain = flops, val_loss, nll, chain
        self._original = (flops, val_loss, nll)
        self.params: dict = {}
        self.feature: Callable = lambda c: np.zeros_like(np.asarray(c, dtype=float))
        self.fit(np.ones(len(flops), dtype=bool))

    def fit_on(self, flops: np.ndarray, val_loss: np.ndarray, nll: np.ndarray) -> None:
        """Refit the stages on another frontier (a bootstrap resample); `restore` undoes it."""
        self.flops, self.val_loss, self.nll = flops, val_loss, nll
        self.fit(np.ones(len(flops), dtype=bool))

    def restore(self) -> None:
        self.flops, self.val_loss, self.nll = self._original
        self.fit(np.ones(len(self.flops), dtype=bool))

    def fit(self, mask: np.ndarray) -> None:
        flops = self.flops[mask]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if self.chain == "val":
                self.params["val"], self.feature = fit_loss_law(flops, self.val_loss[mask])
            elif self.chain == "nll":
                self.params["nll"], self.feature = fit_loss_law(flops, self.nll[mask])
            else:
                val_params, val_law = fit_loss_law(flops, self.val_loss[mask])
                e_val = float(np.exp(val_params[0]))
                stack = np.column_stack([val_law(flops), np.full(len(flops), e_val)])
                l2l_params, _ = fit_parametric_form(
                    func_form=loss_to_loss,
                    X_data=stack,
                    y_data=np.log(self.nll[mask]),
                    initial_grid=list(product(*L2L_GRID)),
                    bounds=[(np.log(1e-4), np.log(self.nll[mask].min())), (-20.0, 20.0), (0.05, 10.0)],
                    delta=HUBER_DELTA,
                    use_scipy=True,
                )
                self.params["val"], self.params["l2l"] = val_params, l2l_params

                def feature(c: np.ndarray, _law=val_law, _p=l2l_params, _e=e_val) -> np.ndarray:
                    c = np.asarray(c, dtype=float)
                    stack = np.column_stack([_law(c), np.full(c.shape[0], _e)])
                    return loss_to_loss(stack, _p, return_log_loss=False)

                self.feature = feature

    def describe(self) -> str:
        parts = []
        if "val" in self.params:
            p = self.params["val"]
            parts.append(f"L_val: E={np.exp(p[0]):.4f}, alpha={p[2]:.3f}")
        if "nll" in self.params:
            p = self.params["nll"]
            parts.append(f"L_nll: E={np.exp(p[0]):.4f}, alpha={p[2]:.3f}")
        if "l2l" in self.params:
            p = self.params["l2l"]
            parts.append(f"l2l: E_real={np.exp(p[0]):.4f}, K={np.exp(p[1]):.3g}, kappa={p[2]:.3f}")
        return "; ".join(parts)


# --- registry -----------------------------------------------------------------
@dataclass
class FormSpec:
    """One candidate functional form for the ROC AUC / FLOPs frontier.

    `to_fit_space` maps observed ROC AUC into the space the form is fitted in and
    `to_auc` maps a prediction back. `ceiling_index` names the parameter carrying the
    irreducible term, with `to_ceiling` / `from_ceiling` converting it to a ROC AUC
    ceiling -- that pair is what lets `profile_ceiling` clamp the ceiling and refit.
    """

    name: str
    func: Callable
    param_names: list[str]
    initial_grid: list[list[float]]
    round: int = 1
    to_fit_space: Callable[[np.ndarray], np.ndarray] = lambda y: y
    to_auc: Callable[[np.ndarray], np.ndarray] = lambda y: y
    fit_space_label: str = "ROC AUC"
    # `fit_parametric_form` always calls the form with return_log_loss=True, so predictions
    # must be made the same way for the gap forms; the AUC-space forms ignore the flag.
    fit_space_is_log: bool = False
    bounds: list[tuple] | None = None
    ceiling_index: int | None = None
    to_ceiling: Callable[[float], float] = lambda v: v
    from_ceiling: Callable[[float], float] = lambda c: c
    fixed_ceiling: float | None = None
    # Best observed ROC AUC that the ceiling bounds were built from; see `bounds_for`.
    bound_auc_max: float = float("nan")
    note: str = ""
    color: str = "#2a78d6"
    linestyle: object = "-"
    describe: Callable[[np.ndarray], str] | None = None
    # Earlier stages of a staged form; refitted on the training subset before every fit.
    stage: StageContext | None = None
    result: dict = field(default_factory=dict)

    def refit_stages(self, mask: np.ndarray | None) -> None:
        if self.stage is not None:
            self.stage.fit(np.ones(len(self.stage.flops), dtype=bool) if mask is None else mask)

    def ceiling(self, params: np.ndarray) -> float:
        if self.ceiling_index is None:
            return float(self.fixed_ceiling) if self.fixed_ceiling is not None else float("nan")
        return float(self.to_ceiling(params[self.ceiling_index]))

    @property
    def has_irreducible(self) -> bool:
        return self.ceiling_index is not None


def _ceiling_from_log_gap(log_e: float) -> float:
    return 1.0 - np.exp(log_e)


def _log_gap_from_ceiling(ceiling: float) -> float:
    return float(np.log(max(1.0 - ceiling, 1e-12)))


def _thin(grid: list, limit: int = MAX_STARTS) -> list:
    """Deterministically thin an initial grid to at most `limit` starts."""
    if len(grid) <= limit:
        return grid
    return grid[:: int(np.ceil(len(grid) / limit))]


# --- descriptions -------------------------------------------------------------
def _d_sigmoid(p):
    a_max, log_x0, k, a_min = p
    return (
        f"ceiling={a_max:.4f}, floor={a_min:.4f}, midpoint 1e{log_x0 / LN10:.2f} FLOPs, steepness {k * LN10:.3f}/decade"
    )


def _d_linear(p):
    return f"AUC = {p[0]:.4f} + {p[1]:.4f} log C  (no ceiling)"


def _d_powerlaw(p):
    return f"gap = {np.exp(p[0]):.4g} * C^{p[1]:.4f}, no irreducible term"


def _d_l0(p):
    return f"irreducible gap E={np.exp(p[0]):.4g}, gap exponent {p[2]:.4f}"


def _d_m4(p):
    return f"irreducible gap E={np.exp(p[0]):.4g}, exponent c={p[2]:.4f}, alpha={p[3]:.4f}"


def _d_m4_free(p):
    return f"irreducible gap E={np.exp(p[0]):.4g}, exponent c={p[2]:.4f}, alpha={p[3]:.4f}, floor AUC={p[4]:.4f}"


def _d_bnsl(p):
    return (
        f"E={np.exp(p[0]):.4g}, exponent -{p[2]:.4f} -> -{p[2] + p[3]:.4f}, "
        f"break 1e{p[4] / LN10:.2f}, sharpness {np.exp(p[5]):.3g}"
    )


def _d_bnsl2(p):
    return (
        f"E={np.exp(p[0]):.4g}, exponent -{p[2]:.4f} -> -{p[2] + p[3]:.4f} -> "
        f"-{p[2] + p[3] + p[6]:.4f}, breaks 1e{p[4] / LN10:.2f} and 1e{p[7] / LN10:.2f}"
    )


def _d_step(p):
    return (
        f"E={np.exp(p[0]):.4g}, exponent {p[2]:.4f}, step {p[3]:.3f} in log-gap at "
        f"1e{p[4] / LN10:.2f} FLOPs (factor {np.exp(p[3]):.2f}), sharpness {p[5]:.3g}"
    )


def _d_step2(p):
    return (
        f"E={np.exp(p[0]):.4g}, exponent {p[2]:.4f}, steps {p[3]:.3f}@1e{p[4] / LN10:.2f} "
        f"and {p[6]:.3f}@1e{p[7] / LN10:.2f}"
    )


def _d_two_stage_logistic(p):
    return f"ceiling={p[0]:.4f}, floor={p[1]:.4f}, link steepness k={p[2]:.4f}, midpoint loss={p[3]:.4f}"


def _d_two_stage_exp(p):
    return f"irreducible gap E={np.exp(p[0]):.4g}, amplitude k={p[1]:.4f}, decay gamma={p[2]:.4f}"


def _d_bounded_pl(p):
    plateau = np.exp(p[0]) + np.exp(p[1])
    return f"E={np.exp(p[0]):.4g}, plateau gap {plateau:.3f}, C0=1e{p[2] / LN10:.2f}, exponent {p[3]:.3f}"


def _d_rosenfeld(p):
    return f"E={np.exp(p[0]):.4g}, a={np.exp(p[1]):.3g}, exponent {p[2]:.3f}, eta={np.exp(p[3]):.3g}"


def _d_bounded_sigmoid(p):
    midpoint = (np.log(C_REF) + p[1] / p[2]) / LN10
    return f"E={np.exp(p[0]):.4g}, midpoint 1e{midpoint:.2f}, steepness {p[2] * LN10:.3f}/decade"


def _d_weibull(p):
    extra = f", floor AUC={p[3]:.3f}" if len(p) > N_WEIBULL_FIXED_FLOOR else ""
    return f"E={np.exp(p[0]):.4g}, a={np.exp(p[1]):.3g}, shape d={p[2]:.3f}{extra}"


def _d_logit_pl(p):
    return f"ceiling={expit(p[0]):.4f}, amplitude {np.exp(p[1]):.3g} (logit units at 1e13), exponent {p[2]:.3f}"


# --- initial grids ------------------------------------------------------------
# The notebook writes the sigmoid midpoint grid as log10(x0) but the form consumes
# natural logs, so midpoints/steepnesses are given in log10/per-decade units and converted.
SIGMOID_GRID = [
    [0.87, 0.9, 0.95, 0.99],
    [m * LN10 for m in (11, 12, 13, 14, 15, 16)],
    [s / LN10 for s in (0.5, 1.0, 2.0, 4.0, 8.0)],
    [0.45, 0.5],
]
SIGMOID_FREE_GRID = [[0.85, 0.9, 0.95, 0.99, 0.999], *SIGMOID_GRID[1:]]
LINEAR_GRID = [[0.0, 0.5], [0.0, 0.01, 0.05]]
L0_GRID = [[0, 0.1, 0.5], [0.1, 1, 2, 5, 10, 50, 100], [-1, -0.5, 0, 0.5, 1]]
POWERLAW_GAP_GRID = [[0.0, 1.0, 2.0, 3.0, 4.0], [-0.5, -0.25, -0.13, -0.05]]
LOG_E_GRID = [np.log(g) for g in (1e-5, 0.02, 0.1)]
M4_GRID = [LOG_E_GRID, [-2.0, 0.0, 2.0], [-0.5, -0.2, -0.05], [0.5, 1.0, 2.0, 4.0]]
M4_FREE_GRID = [LOG_E_GRID, [-2.0, 0.0, 2.0], [-0.5, -0.2, -0.05], [0.5, 1.0, 2.0, 4.0], [0.5, 0.55]]
BNSL_GRID = [
    LOG_E_GRID,
    [0.0, 2.0, 4.0],
    [0.02, 0.05, 0.1],
    [0.05, 0.2, 0.5],
    [m * LN10 for m in (12.5, 13.0, 13.5)],
    [np.log(f) for f in (0.05, 0.2, 1.0)],
]
BNSL2_GRID = [
    LOG_E_GRID,
    [0.0, 3.0],
    [0.02, 0.08],
    [0.3, 1.0],
    [m * LN10 for m in (12.7, 13.0)],
    [np.log(f) for f in (0.05, 0.3)],
    [-0.9, -0.3],
    [m * LN10 for m in (13.2, 13.6)],
    [np.log(f) for f in (0.05, 0.3)],
]
STEPPED_GRID = [
    LOG_E_GRID,
    [0.0, 2.0, 4.0],
    [-0.2, -0.1, -0.05],
    [0.3, 0.5, 1.0],
    [m * LN10 for m in (12.7, 12.9, 13.1)],
    [1.0, 3.0, 10.0],
]
STEPPED2_GRID = [
    LOG_E_GRID,
    [0.0, 3.0],
    [-0.15, -0.05],
    [0.3, 0.7],
    [m * LN10 for m in (12.9,)],
    [3.0, 10.0],
    [0.1, 0.4],
    [m * LN10 for m in (11.5, 12.0)],
    [1.0, 5.0],
]
TWO_STAGE_LOGISTIC_GRID = [[0.87, 0.92, 0.99], [0.45, 0.5], [0.5, 2.0, 5.0], [0.8, 1.2, 2.0]]
TWO_STAGE_EXP_GRID = [LOG_E_GRID, [0.1, 0.3, 0.5, 1.0], [0.1, 0.5, 1.0, 3.0]]
# Real NLL on the frontier runs ~0.95 -> 0.35, so the link midpoints sit lower than for val loss.
NLL_LOGISTIC_GRID = [[0.87, 0.92, 0.99], [0.45, 0.5], [2.0, 5.0, 15.0], [0.3, 0.5, 0.8]]
NLL_EXP_GRID = [LOG_E_GRID, [0.1, 0.3, 0.5, 1.0], [0.5, 1.0, 3.0, 8.0]]
BOUNDED_PL_GRID = [
    LOG_E_GRID,
    [np.log(c) for c in (0.3, 0.45)],
    [m * LN10 for m in (11.0, 12.5, 14.0)],
    [0.1, 0.3, 1.0],
]
ROSENFELD_GRID = [
    LOG_E_GRID,
    [np.log(a) for a in (0.1, 1.0, 10.0)],
    [0.2, 0.5, 1.0],
    [np.log(h) for h in (0.1, 1.0, 10.0)],
]
BOUNDED_SIGMOID_GRID = [LOG_E_GRID, [-2.0, 0.0, 2.0], [0.5, 1.0, 2.0]]
WEIBULL_GRID = [LOG_E_GRID, [-1.0, 0.0, 1.0], [0.2, 0.5, 1.0]]
N_WEIBULL_FIXED_FLOOR = 3
LOGIT_PL_GRID = [[_logit(c) for c in (0.87, 0.9, 0.95)], [-1.0, 0.0, 1.0], [0.1, 0.3, 1.0]]

GAP_KWARGS = dict(
    to_fit_space=lambda y: np.log(1.0 - y),
    to_auc=lambda y: 1.0 - np.exp(y),
    fit_space_label="log(1 - ROC AUC)",
    fit_space_is_log=True,
    ceiling_index=0,
    to_ceiling=_ceiling_from_log_gap,
    from_ceiling=_log_gap_from_ceiling,
)
FREE = (None, None)


def anchor_gap_ceiling(func: Callable, ceiling: float) -> Callable:
    """Hold a gap form's irreducible term fixed at `ceiling`, freeing the rest.

    Rounds 2 and 3 agreed on the ceiling (0.884-0.887) across every two-break variant,
    while the reductions that saved parameters all lost extrapolation accuracy. So round 4
    takes the opposite route: treat the identified ceiling as known, and spend the saved
    parameter on the shape instead.
    """
    log_e = _log_gap_from_ceiling(ceiling)

    def form(data, params, return_log_loss: bool = True):
        return func(data, [log_e, *params], return_log_loss=return_log_loss)

    return form


def anchor_sigmoid_ceiling(ceiling: float) -> Callable:
    """The same, for the sigmoid, whose ceiling is a ROC AUC rather than a log-gap."""

    def form(data, params, return_log_loss: bool = True):
        return sigmoid_functional_form(data, [ceiling, *params], return_log_loss=return_log_loss)

    return form


def _bnsl2_bounds(names: list[str], ceiling_bound: tuple) -> list[tuple]:
    table = {
        "log_e": ceiling_bound,
        "log_b": (-20.0, 20.0),
        "c0": (0.0, 5.0),
        "c1": (0.0, 5.0),
        "c2": (-5.0, 5.0),
        "log_d1": (20.0, 60.0),
        "log_d2": (20.0, 60.0),
        "log_f": (np.log(1e-3), np.log(10.0)),
        "log_f1": (np.log(1e-3), np.log(10.0)),
        "log_f2": (np.log(1e-3), np.log(10.0)),
    }
    return [table[n] for n in names]


def _bnsl2_grid(names: list[str]) -> list[list[float]]:
    table = {
        "log_e": LOG_E_GRID,
        "log_b": [-1.0, 0.0, 2.0],
        "c0": [0.02, 0.08],
        "c1": [0.3, 1.0],
        "c2": [-0.9, -0.3],
        "log_d1": [m * LN10 for m in (12.7, 13.0)],
        "log_d2": [m * LN10 for m in (13.2, 13.6)],
        "log_f": [np.log(f) for f in (0.05, 0.3)],
        "log_f1": [np.log(f) for f in (0.05, 0.3)],
        "log_f2": [np.log(f) for f in (0.05, 0.3)],
    }
    return [table[n] for n in names]


def _bnsl2_spec(name: str, note: str, ceiling_bound: tuple, **kwargs) -> FormSpec:
    func, names = make_bnsl_2break(**kwargs)
    return FormSpec(
        name=name,
        func=func,
        param_names=names,
        initial_grid=_bnsl2_grid(names),
        round=3,
        bounds=_bnsl2_bounds(names, ceiling_bound),
        note=note,
        describe=_d_bnsl2_named(names),
        **GAP_KWARGS,
    )


def fit_val_loss_law(flops: np.ndarray, val_loss: np.ndarray) -> tuple[np.ndarray, Callable]:
    """Stage 1 of the two-stage forms: L(C) = E_L + A * C^-alpha.

    Reuses `functional_form_L0`, so the returned parameters are [l0, a, b] with the
    irreducible loss E_L = exp(l0). Returns the parameters and a callable L(C).
    """
    params, _ = fit_parametric_form(
        func_form=functional_form_L0,
        X_data=flops,
        y_data=np.log(val_loss),
        initial_grid=list(product(*L0_GRID)),
        bounds=None,
        delta=HUBER_DELTA,
        use_scipy=True,
    )

    def loss_law(c: np.ndarray) -> np.ndarray:
        return functional_form_L0(np.asarray(c, dtype=float), params, return_log_loss=False)

    return params, loss_law


def build_forms(
    auc_max: float,
    loss_law: Callable | None = None,
    flops: np.ndarray | None = None,
    val_loss: np.ndarray | None = None,
    nll: np.ndarray | None = None,
) -> list[FormSpec]:
    """The registry. Add a form by appending one FormSpec and giving it a round.

    Staged forms get a `StageContext` built from (flops, val_loss, nll) and refit their earlier
    stages inside every holdout. `loss_law` is kept for callers that pass a fixed stage-1 law
    (the rounds-1-4 behaviour); when the arrays are given it is ignored.
    """
    ceiling_bound = (MIN_LOG_GAP, _log_gap_from_ceiling(auc_max))
    auc_ceiling_bound = (auc_max, 1.0)
    staged = flops is not None and val_loss is not None and nll is not None

    def stage(chain: str) -> StageContext | None:
        return StageContext(flops, val_loss, nll, chain) if staged else None

    def feature_of(ctx: StageContext | None, fallback: Callable | None) -> Callable:
        if ctx is not None:
            # Read `ctx.feature` at call time, since the stages are refitted per holdout.
            return lambda c: ctx.feature(c)  # noqa: PLW0108
        if fallback is None:
            raise ValueError("staged forms need either (flops, val_loss, nll) or a loss_law")
        return fallback

    val_ctx_logistic, val_ctx_exp = stage("val"), stage("val")
    nll_ctx_logistic, nll_ctx_exp = stage("nll"), stage("nll")
    l2l_ctx_logistic, l2l_ctx_exp = stage("l2l"), stage("l2l")
    gap_no_ceiling = {k: v for k, v in GAP_KWARGS.items() if k not in ("ceiling_index", "to_ceiling", "from_ceiling")}

    forms = [
        # -- round 1: published baselines ------------------------------------
        FormSpec(
            name="linear in log FLOPs",
            func=linear_in_log_flops,
            param_names=["a", "b"],
            initial_grid=LINEAR_GRID,
            round=1,
            note="reference, no ceiling",
            describe=_d_linear,
        ),
        FormSpec(
            name="power law on gap",
            func=powerlaw_gap,
            param_names=["a", "b"],
            initial_grid=POWERLAW_GAP_GRID,
            round=1,
            bounds=[(-20.0, 20.0), (-5.0, 0.0)],
            fixed_ceiling=1.0,
            note="Kaplan; reference, no irreducible term",
            describe=_d_powerlaw,
            to_fit_space=lambda y: np.log(1.0 - y),
            to_auc=lambda y: 1.0 - np.exp(y),
            fit_space_label="log(1 - ROC AUC)",
            fit_space_is_log=True,
        ),
        FormSpec(
            name="sigmoid (ceiling free)",
            func=sigmoid_functional_form,
            param_names=["a_max", "log_x0", "k", "a_min"],
            initial_grid=SIGMOID_FREE_GRID,
            round=1,
            ceiling_index=0,
            note="the notebook's fit; ceiling left unconstrained",
            describe=_d_sigmoid,
        ),
        FormSpec(
            name="sigmoid (M4, a=1)",
            func=sigmoid_functional_form,
            param_names=["a_max", "log_x0", "k", "a_min"],
            initial_grid=SIGMOID_GRID,
            round=1,
            bounds=[auc_ceiling_bound, FREE, FREE, FREE],
            ceiling_index=0,
            note="Alabdulmohsin M4 at alpha=1, ceiling >= best observed",
            describe=_d_sigmoid,
        ),
        FormSpec(
            name="power law + irreducible",
            func=functional_form_L0,
            param_names=["l0", "a", "b"],
            initial_grid=L0_GRID,
            round=1,
            bounds=[ceiling_bound, FREE, FREE],
            note="Hoffmann / Chinchilla, on the gap",
            describe=_d_l0,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="M4 (free alpha)",
            func=m4_gap,
            param_names=["log_e", "log_b", "c", "alpha"],
            initial_grid=M4_GRID,
            round=1,
            bounds=[ceiling_bound, (-20.0, 20.0), (-5.0, 0.0), (0.05, 10.0)],
            note="Alabdulmohsin, floor pinned at chance AUC",
            describe=_d_m4,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="BNSL (1 break)",
            func=bnsl_gap,
            param_names=["log_e", "log_b", "c0", "c1", "log_d", "log_f"],
            initial_grid=BNSL_GRID,
            round=1,
            bounds=[ceiling_bound, (-20.0, 20.0), (0.0, 5.0), (0.0, 5.0), (20.0, 60.0), (np.log(1e-3), np.log(10.0))],
            note="Caballero",
            describe=_d_bnsl,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="two-stage, logistic link",
            func=make_two_stage_logistic(feature_of(val_ctx_logistic, loss_law)),
            param_names=["ceiling", "a_min", "k", "l_mid"],
            initial_grid=TWO_STAGE_LOGISTIC_GRID,
            round=1,
            bounds=[auc_ceiling_bound, (0.0, 0.6), (0.0, 50.0), (0.0, 10.0)],
            ceiling_index=0,
            note="Owen / Ruan: predict the metric from the fitted validation loss",
            describe=_d_two_stage_logistic,
            stage=val_ctx_logistic,
        ),
        FormSpec(
            name="two-stage, exponential link",
            func=make_two_stage_exponential(feature_of(val_ctx_exp, loss_law)),
            param_names=["log_e", "k", "gamma"],
            initial_grid=TWO_STAGE_EXP_GRID,
            round=1,
            bounds=[ceiling_bound, (0.0, 10.0), (0.001, 20.0)],
            note="Gadre Eq. 5: gap exponential in the fitted validation loss",
            describe=_d_two_stage_exp,
            stage=val_ctx_exp,
            **GAP_KWARGS,
        ),
        # -- round 2: more than one transition, and a fitted floor -------------
        FormSpec(
            name="M4 (free alpha + floor)",
            func=m4_gap_free_floor,
            param_names=["log_e", "log_b", "c", "alpha", "floor_auc"],
            initial_grid=M4_FREE_GRID,
            round=2,
            bounds=[ceiling_bound, (-20.0, 20.0), (-5.0, 0.0), (0.05, 10.0), (0.4, float(auc_max) - FLOOR_MARGIN)],
            note="M4 with the starting point fitted rather than pinned at chance",
            describe=_d_m4_free,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="step + irreducible",
            func=stepped_powerlaw_gap,
            param_names=["log_e", "a", "b", "d", "log_x0", "k"],
            initial_grid=STEPPED_GRID,
            round=2,
            bounds=[ceiling_bound, (-20.0, 20.0), (-5.0, 0.0), (0.0, 10.0), (20.0, 60.0), (1e-3, 100.0)],
            note="power law interrupted by one transient step",
            describe=_d_step,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="BNSL (2 breaks)",
            func=bnsl_gap_2break,
            param_names=["log_e", "log_b", "c0", "c1", "log_d1", "log_f1", "c2", "log_d2", "log_f2"],
            initial_grid=BNSL2_GRID,
            round=2,
            bounds=[
                ceiling_bound,
                (-20.0, 20.0),
                (0.0, 5.0),
                (0.0, 5.0),
                (20.0, 60.0),
                (np.log(1e-3), np.log(10.0)),
                (-5.0, 5.0),
                (20.0, 60.0),
                (np.log(1e-3), np.log(10.0)),
            ],
            note="Caballero at n=2: shallow -> steep -> shallow",
            describe=_d_bnsl2,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="2 steps + irreducible",
            func=two_stepped_powerlaw_gap,
            param_names=["log_e", "a", "b", "d1", "log_x1", "k1", "d2", "log_x2", "k2"],
            initial_grid=STEPPED2_GRID,
            round=2,
            bounds=[
                ceiling_bound,
                (-20.0, 20.0),
                (-5.0, 0.0),
                (0.0, 10.0),
                (20.0, 60.0),
                (1e-3, 100.0),
                (0.0, 10.0),
                (20.0, 60.0),
                (1e-3, 100.0),
            ],
            note="quantization picture: one jump per capability",
            describe=_d_step2,
            **GAP_KWARGS,
        ),
        # -- round 3: can the identified ceiling survive fewer parameters? -----
        _bnsl2_spec("BNSL 2br (shared sharpness)", "one f for both breaks", ceiling_bound, shared_f=True),
        _bnsl2_spec(
            "BNSL 2br (flat start)", "c0 = 0: gap constant below the first break", ceiling_bound, flat_start=True
        ),
        _bnsl2_spec(
            "BNSL 2br (flat + shared f)", "both reductions at once", ceiling_bound, flat_start=True, shared_f=True
        ),
        _bnsl2_spec(
            "BNSL 2br (flat + sharp)",
            "sharpness fixed, both breaks sharp",
            ceiling_bound,
            flat_start=True,
            fixed_log_f=float(np.log(0.05)),
        ),
        # -- round 4: take the identified ceiling as known ---------------------
        FormSpec(
            name="sigmoid @ anchored ceiling",
            func=anchor_sigmoid_ceiling(ANCHOR_CEILING),
            param_names=["log_x0", "k", "a_min"],
            initial_grid=SIGMOID_GRID[1:],
            round=4,
            fixed_ceiling=ANCHOR_CEILING,
            note=f"ceiling held at {ANCHOR_CEILING}",
            describe=_d_anchored,
        ),
        FormSpec(
            name="BNSL 1br @ anchored ceiling",
            func=anchor_gap_ceiling(bnsl_gap, ANCHOR_CEILING),
            param_names=["log_b", "c0", "c1", "log_d", "log_f"],
            initial_grid=BNSL_GRID[1:],
            round=4,
            bounds=[(-20.0, 20.0), (0.0, 5.0), (0.0, 5.0), (20.0, 60.0), (np.log(1e-3), np.log(10.0))],
            fixed_ceiling=ANCHOR_CEILING,
            note=f"ceiling held at {ANCHOR_CEILING}",
            describe=_d_anchored,
            **{k: v for k, v in GAP_KWARGS.items() if k not in ("ceiling_index", "to_ceiling", "from_ceiling")},
        ),
        FormSpec(
            name="M4 @ anchored ceiling",
            func=anchor_gap_ceiling(m4_gap, ANCHOR_CEILING),
            param_names=["log_b", "c", "alpha"],
            initial_grid=M4_GRID[1:],
            round=4,
            bounds=[(-20.0, 20.0), (-5.0, 0.0), (0.05, 10.0)],
            fixed_ceiling=ANCHOR_CEILING,
            note=f"ceiling held at {ANCHOR_CEILING}",
            describe=_d_anchored,
            **{k: v for k, v in GAP_KWARGS.items() if k not in ("ceiling_index", "to_ceiling", "from_ceiling")},
        ),
        FormSpec(
            name="step @ anchored ceiling",
            func=anchor_gap_ceiling(stepped_powerlaw_gap, ANCHOR_CEILING),
            param_names=["a", "b", "d", "log_x0", "k"],
            initial_grid=STEPPED_GRID[1:],
            round=4,
            bounds=[(-20.0, 20.0), (-5.0, 0.0), (0.0, 10.0), (20.0, 60.0), (1e-3, 100.0)],
            fixed_ceiling=ANCHOR_CEILING,
            note=f"ceiling held at {ANCHOR_CEILING}",
            describe=_d_anchored,
            **gap_no_ceiling,
        ),
        # -- round 5: bounded forms, logit space, and chains through the real NLL -------
        FormSpec(
            name="bounded power law (POW4)",
            func=bounded_powerlaw_gap,
            param_names=["log_e", "log_c", "log_c0", "alpha"],
            initial_grid=BOUNDED_PL_GRID,
            round=5,
            bounds=[ceiling_bound, (-20.0, 0.0), (20.0, 60.0), (0.01, 5.0)],
            note="Arora / LCDB POW4: plateau, then power-law tail above C0",
            describe=_d_bounded_pl,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="Rosenfeld envelope",
            func=rosenfeld_envelope_gap,
            param_names=["log_e", "log_a", "b", "log_eta"],
            initial_grid=ROSENFELD_GRID,
            round=5,
            bounds=[ceiling_bound, (-20.0, 20.0), (0.01, 5.0), (-20.0, 20.0)],
            note="Rosenfeld 2020: power law bounded by chance via |e| / |e - i eta|",
            describe=_d_rosenfeld,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="bounded sigmoid, floor at chance (3p)",
            func=bounded_sigmoid_chance_gap,
            param_names=["log_e", "log_a", "b"],
            initial_grid=BOUNDED_SIGMOID_GRID,
            round=5,
            bounds=[ceiling_bound, (-20.0, 20.0), (0.01, 20.0)],
            note="practical bounded law / M4 at alpha=1: the minimal bounded form",
            describe=_d_bounded_sigmoid,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="Weibull, floor at chance (3p)",
            func=make_weibull_gap(free_floor=False),
            param_names=["log_e", "log_a", "d"],
            initial_grid=WEIBULL_GRID,
            round=5,
            bounds=[ceiling_bound, (-20.0, 20.0), (0.01, 5.0)],
            note="WBL4 with g0 = chance: stretched-exponential approach to the ceiling",
            describe=_d_weibull,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="Weibull (free floor)",
            func=make_weibull_gap(free_floor=True),
            param_names=["log_e", "log_a", "d", "floor_auc"],
            initial_grid=[*WEIBULL_GRID, [0.5, 0.55]],
            round=5,
            bounds=[ceiling_bound, (-20.0, 20.0), (0.01, 5.0), (0.4, float(auc_max) - FLOOR_MARGIN)],
            note="WBL4 with the starting level fitted",
            describe=_d_weibull,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="power law + irreducible (logit space)",
            func=logit_powerlaw,
            param_names=["l_inf", "log_a", "b"],
            initial_grid=LOGIT_PL_GRID,
            round=5,
            bounds=[(float(_logit(auc_max)), float(_logit(MAX_CEILING))), (-20.0, 20.0), (0.01, 5.0)],
            ceiling_index=0,
            to_ceiling=lambda v: float(expit(v)),
            from_ceiling=lambda c: float(_logit(c)),
            to_fit_space=_logit,
            to_auc=expit,
            fit_space_label="logit(ROC AUC)",
            fit_space_is_log=True,
            note="L0 form on logit(AUC): bounded analogue of the log-accuracy law",
            describe=_d_logit_pl,
        ),
        FormSpec(
            name="two-stage via real NLL, logistic link",
            func=make_two_stage_logistic(feature_of(nll_ctx_logistic, None if staged else loss_law)),
            param_names=["ceiling", "a_min", "k", "nll_mid"],
            initial_grid=NLL_LOGISTIC_GRID,
            round=5,
            bounds=[auc_ceiling_bound, (0.0, 0.6), (0.0, 100.0), (0.0, 5.0)],
            ceiling_index=0,
            note="stage 1 = L0 law on real NLL; stage 2 = logistic link",
            describe=_d_two_stage_logistic,
            stage=nll_ctx_logistic,
        ),
        FormSpec(
            name="two-stage via real NLL, exponential link",
            func=make_two_stage_exponential(feature_of(nll_ctx_exp, None if staged else loss_law)),
            param_names=["log_e", "k", "gamma"],
            initial_grid=NLL_EXP_GRID,
            round=5,
            bounds=[ceiling_bound, (0.0, 10.0), (0.001, 50.0)],
            note="stage 1 = L0 law on real NLL; stage 2 = Gadre exponential link",
            describe=_d_two_stage_exp,
            stage=nll_ctx_exp,
            **GAP_KWARGS,
        ),
        FormSpec(
            name="loss-to-loss chain, logistic link",
            func=make_two_stage_logistic(feature_of(l2l_ctx_logistic, None if staged else loss_law)),
            param_names=["ceiling", "a_min", "k", "nll_mid"],
            initial_grid=NLL_LOGISTIC_GRID,
            round=5,
            bounds=[auc_ceiling_bound, (0.0, 0.6), (0.0, 100.0), (0.0, 5.0)],
            ceiling_index=0,
            note="Brandfonbrener: L_val(C) -> L_nll = E + K (L_val - E_val)^kappa -> logistic link",
            describe=_d_two_stage_logistic,
            stage=l2l_ctx_logistic,
        ),
        FormSpec(
            name="loss-to-loss chain, exponential link",
            func=make_two_stage_exponential(feature_of(l2l_ctx_exp, None if staged else loss_law)),
            param_names=["log_e", "k", "gamma"],
            initial_grid=NLL_EXP_GRID,
            round=5,
            bounds=[ceiling_bound, (0.0, 10.0), (0.001, 50.0)],
            note="Brandfonbrener chain into the Gadre exponential link",
            describe=_d_two_stage_exp,
            stage=l2l_ctx_exp,
            **GAP_KWARGS,
        ),
    ]
    for spec, color, dash in zip(forms, cycle(PALETTE), cycle(DASHES)):
        spec.color = color
        spec.bound_auc_max = float(auc_max)
        spec.linestyle = dash
        spec.initial_grid = spec.initial_grid
    return forms


# --- fitting ------------------------------------------------------------------
def bounds_for(spec: FormSpec, auc_train_max: float) -> list[tuple] | None:
    """The form's bounds with every bound tied to the best observed ROC AUC re-tied to the
    best ROC AUC of the points actually being fitted.

    The registry requires the ceiling to lie at or above the best observed point, and the
    fitted floors 0.05 below it. Built from the whole frontier, those bounds leak the test
    points into every holdout refit: a refit on points up to 3e13 FLOPs was forced to put its
    ceiling at or above the best point at 3e16, and most forms then sat exactly on that bound,
    so the bound did the extrapolating. Rounds 1-4 and the first round-5 runs were affected.
    """
    ref = spec.bound_auc_max
    if spec.bounds is None or not np.isfinite(ref) or auc_train_max >= ref - 1e-12:
        return spec.bounds
    out = []
    for i, (lo, hi) in enumerate(spec.bounds):
        new = [lo, hi]
        for j, edge in enumerate((lo, hi)):
            if edge is None:
                continue
            if i == spec.ceiling_index and np.isclose(spec.to_ceiling(edge), ref, rtol=0, atol=1e-9):
                new[j] = spec.from_ceiling(auc_train_max)
            elif np.isclose(edge, ref - FLOOR_MARGIN, rtol=0, atol=1e-12):
                new[j] = max(auc_train_max - FLOOR_MARGIN, (lo if lo is not None else -np.inf) + 1e-6)
        out.append(tuple(new))
    return out


def _fit_space_weights(spec: FormSpec, flops: np.ndarray) -> np.ndarray | None:
    """Inverse-noise weights in the form's fit space, or None when no noise model is loaded.

    The seed study gives sigma in ROC AUC units per FLOP level. A form fitted in another space
    (log gap, logit) sees that noise scaled by the derivative of its transform, so the residual
    of point i is divided by sigma_i |d fit / d AUC|_i. Weights are normalised to a median of 1,
    which keeps the Huber delta meaning what it meant for the unweighted fits.
    """
    if not NOISE["sigma_auc"]:
        return None
    flops = np.asarray(flops, dtype=float)
    sigma = np.array([NOISE["sigma_auc"][f] for f in flops])
    auc = np.array([NOISE["auc"][f] for f in flops])
    h = 1e-4
    slope = np.abs(spec.to_fit_space(auc + h) - spec.to_fit_space(auc - h)) / (2 * h)
    sigma_fit = np.maximum(sigma * slope, 1e-9)
    weights = 1.0 / sigma_fit
    return weights / np.median(weights)


def _fit(spec: FormSpec, flops, y_fit, initial_grid, bounds):
    weights = _fit_space_weights(spec, flops)
    func, y = spec.func, y_fit
    if weights is not None:
        lookup = dict(zip(np.asarray(flops, dtype=float), weights))

        def func(data, params, return_log_loss: bool = True, _f=spec.func):
            w = np.array([lookup[d] for d in np.asarray(data, dtype=float)])
            return _f(data, params, return_log_loss=return_log_loss) * w

        y = y_fit * weights
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return fit_parametric_form(
            func_form=func,
            X_data=flops,
            y_data=y,
            initial_grid=initial_grid,
            bounds=bounds,
            delta=HUBER_DELTA,
            use_scipy=True,
        )


def _refit_starts(spec: FormSpec) -> list:
    """Starts for the many holdout refits: the initial grid, thinned to REFIT_SEEDS.

    Never the full fit's parameters. Those were fitted on the test points too, and a form
    with more parameters than training points barely moves from its start, so seeding a
    holdout refit with them leaks the answer -- it made BNSL-2 look like the best long-range
    extrapolator in an earlier version of this ladder.
    """
    return [list(g) for g in _thin(list(product(*spec.initial_grid)), REFIT_SEEDS)]


def _predict(spec: FormSpec, params, flops: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return spec.to_auc(spec.func(np.asarray(flops, dtype=float), params, return_log_loss=spec.fit_space_is_log))


def predict_auc(spec: FormSpec, flops: np.ndarray) -> np.ndarray:
    return _predict(spec, spec.result["params"], flops)


def fit_form(spec: FormSpec, flops: np.ndarray, auc: np.ndarray) -> dict:
    """Fit on all points and score in-sample."""
    y_fit = spec.to_fit_space(auc)
    starts = _thin(list(product(*spec.initial_grid)))
    spec.refit_stages(None)
    best_params, best_loss = _fit(spec, flops, y_fit, starts, spec.bounds)

    pred_fit = spec.func(flops, best_params, return_log_loss=spec.fit_space_is_log)
    pred_auc = spec.to_auc(pred_fit)

    spec.result = {
        "params": best_params,
        "n_params": len(best_params),
        "n_starts": len(starts),
        "huber_loss": best_loss,
        "ceiling": spec.ceiling(best_params),
        "r2_auc": r2_score(auc, pred_auc),
        "rmse_auc": float(np.sqrt(np.mean((auc - pred_auc) ** 2))),
        "max_abs_err_auc": float(np.max(np.abs(auc - pred_auc))),
        "residuals_auc": auc - pred_auc,
        "auc_extrapolated": float(_predict(spec, best_params, np.array([EXTRAPOLATION_FLOPS]))[0]),
        "stages": spec.stage.describe() if spec.stage is not None else "",
    }
    return spec.result


def holdout_score(spec: FormSpec, flops: np.ndarray, auc: np.ndarray) -> dict:
    """Refit on the low-compute points only, then score the points above.

    This is the number the search is actually optimising: in-sample R^2 says nothing
    about whether a form extrapolates, which is the only reason to fit one here.

    Two thresholds are used, not one. A single cut at 1e14 leaves four test points that
    all sit in the flattening tail, which rewards any form that saturates; the harder cut
    at 3.2e13 tests whether the form can see the tail coming at all. The headline number
    is the mean of the two.
    """
    starts = _thin(list(product(*spec.initial_grid)))
    per_threshold = {}
    for threshold in HOLDOUT_FLOPS_LIST:
        train = flops <= threshold
        test = ~train
        if test.sum() == 0 or train.sum() < MIN_HOLDOUT_TRAIN:
            continue
        spec.refit_stages(train)
        params, _ = _fit(spec, flops[train], spec.to_fit_space(auc[train]), starts, bounds_for(spec, auc[train].max()))
        pred = _predict(spec, params, flops[test])
        errors = auc[test] - pred
        per_threshold[threshold] = {
            "params": params,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
            "mae": float(np.mean(np.abs(errors))),
            "bias": float(np.mean(errors)),
            "max_abs": float(np.max(np.abs(errors))),
            "ceiling": spec.ceiling(params),
        }
    spec.refit_stages(None)
    if not per_threshold:
        return {}

    maes = [v["mae"] for v in per_threshold.values()]
    widest = per_threshold[min(per_threshold)]  # hardest cut, fewest training points
    spec.result["holdout"] = {
        "per_threshold": per_threshold,
        "mae": float(np.mean(maes)),
        "mae_easy": per_threshold[max(per_threshold)]["mae"],
        "mae_hard": widest["mae"],
        "bias": float(np.mean([v["bias"] for v in per_threshold.values()])),
        "params": per_threshold[max(per_threshold)]["params"],
        "params_hard": widest["params"],
    }
    return spec.result["holdout"]


def ladder_score(spec: FormSpec, flops: np.ndarray, auc: np.ndarray) -> dict:
    """Leave-future-out ladder: fit on the points up to each FLOP level, score everything above.

    One cut rewards whichever form happens to bend at the right place; the ladder asks the
    form to extrapolate from every prefix of the frontier that leaves at least
    MIN_LADDER_TEST test points. The headline is the median forward MAE over cuts, which is
    what the scaling-law literature recommends for a dozen-odd points (Caballero et al.,
    Owen 2024, Choshen et al. 2024).
    """
    levels = np.unique(flops)
    starts = _refit_starts(spec)
    rows = []
    for cut in levels:
        train, test = flops <= cut, flops > cut
        if train.sum() < MIN_HOLDOUT_TRAIN or test.sum() < MIN_LADDER_TEST:
            continue
        spec.refit_stages(train)
        params, _ = _fit(spec, flops[train], spec.to_fit_space(auc[train]), starts, bounds_for(spec, auc[train].max()))
        errors = auc[test] - _predict(spec, params, flops[test])
        rows.append({"cut": float(cut), "n_train": int(train.sum()), "mae": float(np.mean(np.abs(errors)))})
    spec.refit_stages(None)
    if not rows:
        return {}
    maes = np.array([r["mae"] for r in rows])
    spec.result["ladder"] = {"cuts": rows, "median_mae": float(np.median(maes)), "max_mae": float(maes.max())}
    return spec.result["ladder"]


def loro_score(spec: FormSpec, flops: np.ndarray, auc: np.ndarray, runs: np.ndarray) -> dict:
    """Leave-one-run-out: hold out every point of each source run in turn, pool the errors.

    No test point ever shares a learning curve with a training point, which is the leak the
    FLOP-cut holdouts have when a few runs supply most of the frontier.
    """
    starts = _refit_starts(spec)
    errors = []
    for run in np.unique(runs):
        test = runs == run
        train = ~test
        if train.sum() < MIN_HOLDOUT_TRAIN:
            continue
        spec.refit_stages(train)
        params, _ = _fit(spec, flops[train], spec.to_fit_space(auc[train]), starts, bounds_for(spec, auc[train].max()))
        errors.extend((auc[test] - _predict(spec, params, flops[test])).tolist())
    spec.refit_stages(None)
    if not errors:
        return {}
    errors = np.array(errors)
    spec.result["loro"] = {"mae": float(np.mean(np.abs(errors))), "bias": float(np.mean(errors)), "n": len(errors)}
    return spec.result["loro"]


def bootstrap_runs(
    forms: list[FormSpec], table: pd.DataFrame, n_boot: int, topk: int, rng: np.random.Generator
) -> None:
    """Bootstrap over training runs, not frontier points.

    Resampling frontier points cannot see the selection noise of a max over thousands of
    runs; resampling the runs, rebuilding the frontier and refitting can. Each resample
    draws runs with replacement, rebuilds the frontier once, and refits every form with an
    irreducible term from its full-fit parameters plus a thinned grid. Reports percentile
    intervals for the ceiling and the extrapolated ROC AUC. The seed-noise weights, if on,
    are looked up by FLOP level, so they carry over to the resampled frontiers.
    """
    specs = [s for s in forms if s.has_irreducible]
    if not specs or n_boot <= 0:
        return
    positions = table.groupby(level=0).indices
    run_ids = np.array(list(positions))
    starts = {s.name: _bootstrap_starts(s) for s in specs}
    draws: dict[str, list] = {s.name: [] for s in specs}
    far: dict[str, list] = {s.name: [] for s in specs}
    for _ in range(n_boot):
        sample = rng.choice(run_ids, size=len(run_ids), replace=True)
        frontier = frontier_from_table(table.iloc[np.concatenate([positions[r] for r in sample])], topk)
        if len(frontier) < MIN_HOLDOUT_TRAIN:
            continue
        flops = frontier[FLOPS_COLUMN].values.astype(float)
        auc = frontier[LOSS_COLUMN].values.astype(float)
        _extend_noise(flops, auc)
        for spec in specs:
            if spec.stage is not None:
                spec.stage.fit_on(
                    flops, frontier[VAL_LOSS_COLUMN].values.astype(float), frontier[NLL_COLUMN].values.astype(float)
                )
            params, _ = _fit(spec, flops, spec.to_fit_space(auc), starts[spec.name], bounds_for(spec, auc.max()))
            draws[spec.name].append(spec.ceiling(params))
            far[spec.name].append(float(_predict(spec, params, np.array([EXTRAPOLATION_FLOPS]))[0]))
    for spec in specs:
        if spec.stage is not None:
            spec.stage.restore()
        _store_bootstrap(spec, np.array(draws[spec.name]), np.array(far[spec.name]))


def _bootstrap_starts(spec: FormSpec) -> list:
    grid = _thin(list(product(*spec.initial_grid)))
    starts = [list(g) for g in grid[:: max(1, len(grid) // BOOTSTRAP_SEEDS)]]
    return [list(spec.result["params"]), *starts]


def _extend_noise(flops: np.ndarray, auc: np.ndarray) -> None:
    """Give FLOP levels that the original frontier lacked the noise of their nearest level."""
    if not NOISE["sigma_auc"]:
        return
    known = np.array(list(NOISE["sigma_auc"]))
    for f, a in zip(flops, auc):
        nearest = float(known[np.argmin(np.abs(np.log10(known) - np.log10(f)))])
        NOISE["sigma_auc"].setdefault(float(f), NOISE["sigma_auc"][nearest])
        NOISE["auc"].setdefault(float(f), float(a))


def _store_bootstrap(spec: FormSpec, ceilings: np.ndarray, fars: np.ndarray) -> None:
    if ceilings.size == 0:
        return
    spec.result["bootstrap"] = {
        "n": int(ceilings.size),
        "ceiling_lo": float(np.percentile(ceilings, 2.5)),
        "ceiling_hi": float(np.percentile(ceilings, 97.5)),
        "ceiling_median": float(np.median(ceilings)),
        "far_lo": float(np.percentile(fars, 2.5)),
        "far_hi": float(np.percentile(fars, 97.5)),
        "ceilings": ceilings,
    }


def group_holdout_score(spec: FormSpec, flops: np.ndarray, auc: np.ndarray, runs: np.ndarray) -> dict:
    """Hold out an entire training run, so no test point shares a run with a training point.

    The FLOP-threshold holdout above is contaminated: 9 of the 15 Pareto points come from
    just two runs, so cutting on FLOPs usually leaves the test points on the *same*
    learning curve the fit already saw. Predicting later checkpoints of a run you have
    already been shown is interpolation along one curve, not extrapolation to a new,
    larger model. This holds out every point belonging to the run that contains the
    highest-FLOP point, which is the only clean extrapolation test the data allows.
    """
    largest_run = runs[np.argmax(flops)]
    test = runs == largest_run
    train = ~test
    if test.sum() == 0 or train.sum() < MIN_HOLDOUT_TRAIN:
        return {}

    starts = _thin(list(product(*spec.initial_grid)))
    spec.refit_stages(train)
    params, _ = _fit(spec, flops[train], spec.to_fit_space(auc[train]), starts, bounds_for(spec, auc[train].max()))
    errors = auc[test] - _predict(spec, params, flops[test])
    spec.refit_stages(None)
    spec.result["group_holdout"] = {
        "params": params,
        "held_out_run": largest_run,
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "ceiling": spec.ceiling(params),
    }
    return spec.result["group_holdout"]


def holdout_contamination(flops: np.ndarray, runs: np.ndarray) -> dict:
    """Fraction of each FLOP-cut's test points whose run also appears in training."""
    report = {}
    for threshold in HOLDOUT_FLOPS_LIST:
        train, test = flops <= threshold, flops > threshold
        seen = set(runs[train])
        shared = int(sum(r in seen for r in runs[test]))
        report[threshold] = {"n_test": int(test.sum()), "same_run_as_train": shared}
    return report


def profile_ceiling(spec: FormSpec, flops: np.ndarray, auc: np.ndarray) -> dict | None:
    """Refit with the ceiling clamped, across a grid of ceilings.

    A flat profile means the data tolerates any ceiling in that range -- i.e. the
    irreducible loss is not identified, whatever the point estimate says.
    """
    if not spec.has_irreducible:
        return None

    spec.refit_stages(None)
    y_fit = spec.to_fit_space(auc)
    idx = spec.ceiling_index
    base_bounds = list(spec.bounds) if spec.bounds is not None else [FREE] * len(spec.result["params"])
    grid = _thin(list(product(*spec.initial_grid)))
    seeds = [list(spec.result["params"]), *[list(g) for g in grid[:: max(1, len(grid) // PROFILE_SEEDS)]]]

    ceilings = np.linspace(auc.max(), MAX_CEILING, CEILING_GRID_SIZE)
    losses = []
    for ceiling in ceilings:
        value = spec.from_ceiling(ceiling)
        bounds = list(base_bounds)
        bounds[idx] = (value, value)  # equal bounds == parameter held fixed
        inits = [[*seed[:idx], value, *seed[idx + 1 :]] for seed in seeds]
        _, loss = _fit(spec, flops, y_fit, inits, bounds)
        losses.append(loss)

    losses = np.asarray(losses)
    best = float(losses.min())
    tolerated = ceilings[losses <= best * CEILING_TOLERANCE]
    profile = {
        "ceilings": ceilings,
        "losses": losses,
        "best_profiled_ceiling": float(ceilings[int(losses.argmin())]),
        "ceiling_lo": float(tolerated.min()),
        "ceiling_hi": float(tolerated.max()),
        "identified": bool(tolerated.max() < MAX_CEILING and tolerated.min() > ceilings[0]),
    }
    spec.result["profile"] = profile
    return profile


def flops_to_reach(spec: FormSpec, target: float, max_log10: float = 40.0) -> float | None:
    grid = np.logspace(11, max_log10, 20000)
    reached = np.flatnonzero(predict_auc(spec, grid) >= target)
    return float(grid[reached[0]]) if reached.size else None


# --- reporting ----------------------------------------------------------------
def _print_holdouts(r: dict) -> None:
    hold = r.get("holdout", {})
    if hold:
        print(f"  HOLDOUT  mean MAE {hold['mae']:.4f}   (bias {hold['bias']:+.4f})")
        for threshold, res in sorted(hold["per_threshold"].items()):
            print(
                f"    fit <= {threshold:.2e} ({res['n_train']} pts) -> test {res['n_test']} pts: "
                f"MAE {res['mae']:.4f}, bias {res['bias']:+.4f}, ceiling {res['ceiling']:.4f}"
            )
    group = r.get("group_holdout", {})
    if group:
        print(
            f"  GROUP HOLDOUT (leave out run {group['held_out_run']} entirely: "
            f"{group['n_train']} train, {group['n_test']} test)"
        )
        print(f"    MAE {group['mae']:.4f}   bias {group['bias']:+.4f}   ceiling {group['ceiling']:.4f}")
    ladder = r.get("ladder", {})
    if ladder:
        cuts = ", ".join(f"1e{np.log10(c['cut']):.2f}:{c['mae']:.4f}" for c in ladder["cuts"])
        print(f"  LADDER   median MAE {ladder['median_mae']:.4f}   max {ladder['max_mae']:.4f}   [{cuts}]")
    loro = r.get("loro", {})
    if loro:
        print(f"  LEAVE-ONE-RUN-OUT   MAE {loro['mae']:.4f}   bias {loro['bias']:+.4f}   ({loro['n']} points)")
    boot = r.get("bootstrap", {})
    if boot:
        far = f"[{boot['far_lo']:.4f}, {boot['far_hi']:.4f}]"
        print(
            f"  RUN BOOTSTRAP ({boot['n']} resamples): ceiling 95% [{boot['ceiling_lo']:.4f}, "
            f"{boot['ceiling_hi']:.4f}], AUC @ {EXTRAPOLATION_FLOPS:.0e} {far}"
        )


def _print_form(spec: FormSpec, target_auc: float | None) -> float | None:
    """Print one form's results; return the compute needed to reach `target_auc`, if asked."""
    r = spec.result
    profile = r.get("profile")
    print(f"\n=== {spec.name} ===   [{spec.note}]")
    print("  params: " + ", ".join(f"{n}={v:.6g}" for n, v in zip(spec.param_names, r["params"])))
    if spec.describe is not None:
        print(f"  -> {spec.describe(r['params'])}")
    if r.get("stages"):
        print(f"  stages (fit on all points): {r['stages']}")
    _print_holdouts(r)
    print(f"  in-sample R^2 {r['r2_auc']:.4f}   RMSE {r['rmse_auc']:.4f}   max |err| {r['max_abs_err_auc']:.4f}")
    if not spec.has_irreducible:
        print(f"  ceiling : {r['ceiling']:.4f}  (none fitted -- reference form)")
    else:
        verdict = "identified" if profile["identified"] else "NOT identified"
        print(f"  ceiling (irreducible ROC AUC) : {r['ceiling']:.4f}")
        print(
            f"    tolerated range (loss <= {CEILING_TOLERANCE:.2f}x best): "
            f"[{profile['ceiling_lo']:.4f}, {profile['ceiling_hi']:.4f}] -- {verdict}"
        )
    print(f"  ROC AUC at {EXTRAPOLATION_FLOPS:.0e} FLOPs : {r['auc_extrapolated']:.4f}")
    if target_auc is None:
        return None
    reach = flops_to_reach(spec, target_auc)
    pretty = f"1e{np.log10(reach):.1f}" if reach is not None else "never (< 1e40)"
    print(f"  FLOPs to reach ROC AUC {target_auc:.3f} : {pretty}")
    return reach


def report(specs: list[FormSpec], target_auc: float | None) -> pd.DataFrame:
    rows = []
    for spec in specs:
        r = spec.result
        profile = r.get("profile")
        hold = r.get("holdout", {})
        reach = _print_form(spec, target_auc)
        rows.append(
            {
                "form": spec.name,
                "round": spec.round,
                "n params": r["n_params"],
                "holdout MAE": hold.get("mae", np.nan),
                "MAE hard cut": hold.get("mae_hard", np.nan),
                "MAE easy cut": hold.get("mae_easy", np.nan),
                "group MAE": r.get("group_holdout", {}).get("mae", np.nan),
                "ladder median MAE": r.get("ladder", {}).get("median_mae", np.nan),
                "ladder max MAE": r.get("ladder", {}).get("max_mae", np.nan),
                "LORO MAE": r.get("loro", {}).get("mae", np.nan),
                "boot ceiling lo": r.get("bootstrap", {}).get("ceiling_lo", np.nan),
                "boot ceiling hi": r.get("bootstrap", {}).get("ceiling_hi", np.nan),
                "holdout bias": hold.get("bias", np.nan),
                "R2 (in-sample)": r["r2_auc"],
                "RMSE (in-sample)": r["rmse_auc"],
                "ceiling": r["ceiling"],
                "ceiling lo": profile["ceiling_lo"] if profile else np.nan,
                "ceiling hi": profile["ceiling_hi"] if profile else np.nan,
                "ceiling identified": profile["identified"] if profile else False,
                f"AUC @ {EXTRAPOLATION_FLOPS:.0e}": r["auc_extrapolated"],
                "log10 FLOPs to target": np.log10(reach) if reach is not None else np.inf,
            }
        )

    summary = pd.DataFrame(rows).set_index("form")
    rank_by = "ladder median MAE" if summary["ladder median MAE"].notna().any() else "group MAE"
    summary = summary.sort_values(rank_by)
    thresholds = ", ".join(f"{t:.2e}" for t in HOLDOUT_FLOPS_LIST)
    print(f"\n--- summary (ranked by {rank_by}; FLOP cuts at {thresholds} shown for reference) ---")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    return summary


def rank_key(spec: FormSpec) -> float:
    """The number the search ranks by: ladder median MAE, else the group holdout MAE."""
    ladder = spec.result.get("ladder", {})
    if ladder:
        return ladder["median_mae"]
    return spec.result.get("group_holdout", {}).get("mae", np.inf)


# --- plotting -----------------------------------------------------------------
def _despine(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("#9ca3af")


def _ranges(flops: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge = np.log10(flops.max()) + 0.3
    return (
        np.logspace(np.log10(flops.min()) - 0.3, edge, 400),
        np.logspace(edge, np.log10(EXTRAPOLATION_FLOPS) + 0.5, 400),
    )


def _baseline(ax, tabpfn_auc, x_left, label=True):
    if tabpfn_auc is None:
        return
    ax.axhline(tabpfn_auc, color="#6b7280", linewidth=1.2, linestyle=":", zorder=1)
    if label:
        ax.annotate(
            f"TabPFN on TabArena ({tabpfn_auc:.3f})",
            xy=(x_left, tabpfn_auc),
            xytext=(4, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize="x-small",
            color="#6b7280",
        )


def plot_comparison(specs, flops, auc, tabpfn_auc, title):
    x_obs, x_extra = _ranges(flops)
    x_all = np.concatenate([x_obs, x_extra])
    fig, (ax, ax_res) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], figsize=(11, 10), layout="constrained")

    _baseline(ax, tabpfn_auc, x_obs[0])
    ax.axvspan(flops.max(), x_extra[-1], color="#6b7280", alpha=0.06, zorder=0, linewidth=0)
    ax.annotate(
        "extrapolation",
        xy=(np.sqrt(flops.max() * x_extra[-1]), 0.995),
        ha="center",
        va="top",
        fontsize="x-small",
        color="#6b7280",
    )

    # Colour by position in *this* subset: the global assignment repeats once there are
    # more forms than palette slots, which would put two curves in the same hue here.
    for i, spec in enumerate(specs):
        color, dash = PALETTE[i % len(PALETTE)], DASHES[i % len(DASHES)]
        hold = spec.result.get("holdout", {})
        label = f"{spec.name}  (holdout MAE {hold.get('mae', float('nan')):.3f}, ceiling {spec.result['ceiling']:.3f})"
        ax.plot(x_all, predict_auc(spec, x_all), color=color, linestyle=dash, linewidth=2, label=label, zorder=3)
        ax_res.plot(
            flops,
            spec.result["residuals_auc"],
            marker="o",
            markersize=4,
            linewidth=1,
            color=color,
            linestyle=dash,
            markeredgecolor="white",
            markeredgewidth=0.7,
        )

    ax.scatter(flops, auc, marker="x", s=60, linewidth=1.8, color="#111827", label=POINT_LABEL["text"], zorder=4)
    ax.set_xscale("log")
    ax.set_ylabel("real_data/roc_auc")
    ax.set_ylim(0.45, 1.02)
    ax.set_title(title, fontsize="small")
    ax.legend(loc="upper left", fontsize="xx-small", framealpha=0.92)
    _despine(ax)

    ax_res.axhline(0, color="#9ca3af", linewidth=1)
    ax_res.set_ylabel("residual\n(obs - fit)", fontsize="small")
    ax_res.set_xlabel("Training FLOPs")
    _despine(ax_res)
    return fig


def plot_small_multiples(specs, flops, auc, tabpfn_auc):
    """One panel per form, showing the full fit, the holdout fit, and the ceiling band."""
    x_obs, x_extra = _ranges(flops)
    x_all = np.concatenate([x_obs, x_extra])
    ncols = 3
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, figsize=(13, 3.9 * nrows), layout="constrained", squeeze=False
    )

    for ax, spec in zip(axes.flat, specs):
        _baseline(ax, tabpfn_auc, x_obs[0], label=False)
        ax.axvspan(flops.max(), x_extra[-1], color="#6b7280", alpha=0.06, zorder=0, linewidth=0)
        ax.axvspan(min(HOLDOUT_FLOPS_LIST), flops.max(), color="#111827", alpha=0.05, zorder=0, linewidth=0)
        profile = spec.result.get("profile")
        if profile is not None:
            ax.axhspan(profile["ceiling_lo"], profile["ceiling_hi"], color=spec.color, alpha=0.1, zorder=0, linewidth=0)
        if spec.has_irreducible:
            ax.axhline(spec.result["ceiling"], color=spec.color, linewidth=1, linestyle=":", alpha=0.8, zorder=1)
        hold = spec.result.get("holdout", {})
        if hold:
            for key, alpha in (("params", 0.9), ("params_hard", 0.5)):
                ax.plot(
                    x_all,
                    _predict(spec, hold[key], x_all),
                    color="#6b7280",
                    linewidth=1.4,
                    linestyle=(0, (2, 2)),
                    alpha=alpha,
                    zorder=2,
                )
        ax.plot(x_all, predict_auc(spec, x_all), color=spec.color, linestyle=spec.linestyle, linewidth=2, zorder=3)
        ax.scatter(flops, auc, marker="x", s=30, linewidth=1.2, color="#111827", zorder=4)
        ax.set_xscale("log")
        ax.set_ylim(0.45, 1.02)
        ax.set_title(
            f"{spec.name}\nholdout MAE {hold.get('mae', float('nan')):.3f}, "
            f"{spec.result['n_params']} params, ceiling {spec.result['ceiling']:.3f}",
            fontsize="xx-small",
        )
        _despine(ax)

    for ax in axes.flat[len(specs) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Training FLOPs", fontsize="small")
    for ax in axes[:, 0]:
        ax.set_ylabel("real_data/roc_auc", fontsize="small")
    fig.suptitle(
        "colour = fit on all points; grey dashed = holdout refits (dark: <= 1e14, light: <= 3.2e13); "
        "dotted line = ceiling, band = ceilings the data tolerates",
        fontsize="small",
    )
    return fig


def plot_gap(specs, flops, auc, title):
    x_obs, x_extra = _ranges(flops)
    x_all = np.concatenate([x_obs, x_extra])
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    for i, spec in enumerate(specs):
        gap = 1.0 - predict_auc(spec, x_all)
        ax.plot(
            x_all,
            np.clip(gap, 1e-12, None),
            color=PALETTE[i % len(PALETTE)],
            linestyle=DASHES[i % len(DASHES)],
            linewidth=2,
            label=spec.name,
        )
    ax.scatter(flops, 1.0 - auc, marker="x", s=60, linewidth=1.8, color="#111827", label="Pareto points", zorder=4)
    ax.axvspan(flops.max(), x_extra[-1], color="#6b7280", alpha=0.06, zorder=0, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.0)
    ax.set_xlabel("Training FLOPs")
    ax.set_ylabel("1 - ROC AUC  (gap)")
    ax.set_title(title, fontsize="small")
    ax.legend(loc="lower left", fontsize="xx-small", framealpha=0.92)
    _despine(ax)
    return fig


def plot_ceiling_profile(specs, auc_max):
    fig, ax = plt.subplots(figsize=(11, 6.5), layout="constrained")
    for spec in specs:
        profile = spec.result.get("profile")
        if profile is None:
            continue
        ax.plot(
            profile["ceilings"],
            profile["losses"] / profile["losses"].min(),
            color=spec.color,
            linestyle=spec.linestyle,
            linewidth=2,
            label=spec.name,
        )
    ax.axhline(CEILING_TOLERANCE, color="#6b7280", linewidth=1.2, linestyle=":")
    ax.annotate(
        f"{CEILING_TOLERANCE:.0%} of best loss -- below this line the ceiling is tolerated",
        xy=(auc_max, CEILING_TOLERANCE),
        xytext=(4, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize="x-small",
        color="#6b7280",
    )
    ax.set_xlabel("ceiling: irreducible ROC AUC (clamped)")
    ax.set_ylabel("Huber loss / best")
    ax.set_ylim(0.95, 3.0)
    ax.set_title("How well does the data pin down the irreducible loss?", fontsize="small")
    ax.legend(loc="upper right", fontsize="xx-small", framealpha=0.92)
    _despine(ax)
    return fig


def plot_val_loss_stage(flops, val_loss, params):
    """Stage 1 of the two-stage forms, with its own irreducible loss marked."""
    x_obs, x_extra = _ranges(flops)
    x_all = np.concatenate([x_obs, x_extra])
    irreducible = float(np.exp(params[0]))
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.scatter(flops, val_loss, marker="x", s=60, linewidth=1.8, color="#111827", label="Pareto points")
    ax.plot(
        x_all,
        functional_form_L0(x_all, params, return_log_loss=False),
        color=PALETTE[0],
        linewidth=2,
        label=f"$L = E_L + A\\,C^{{-\\alpha}}$, $\\alpha$={-params[2]:.3f}",
    )
    ax.axhline(
        irreducible, color=PALETTE[0], linestyle=":", linewidth=1.2, label=f"irreducible loss $E_L$ = {irreducible:.4f}"
    )
    ax.axvspan(flops.max(), x_extra[-1], color="#6b7280", alpha=0.06, zorder=0, linewidth=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training FLOPs")
    ax.set_ylabel(VAL_LOSS_COLUMN)
    ax.set_title("Two-stage forms, stage 1: prior validation loss vs compute", fontsize="small")
    ax.legend(loc="upper right", fontsize="x-small", framealpha=0.92)
    _despine(ax)
    return fig


# --- data ---------------------------------------------------------------------
POINT_LABEL = {"text": "Pareto points (big prior)"}


def frontier_from_table(table: pd.DataFrame, topk: int = 1) -> pd.DataFrame:
    """Compute -> ROC AUC frontier from a run table (one row per run and FLOP level).

    topk = 1: the running max over runs, the construction of rounds 1-4.
    topk > 1: the mean of the k best runs per FLOP level, which dampens the winner's curse
        of a max over thousands of noisy runs; levels with fewer than MIN_RUNS_PER_TOPK * k
        runs are dropped. The row is indexed by the best run of the level.
    """
    rows = []
    for _level, group in table.groupby(FLOPS_COLUMN):
        if topk <= 1:
            # positional, since a bootstrap resample carries duplicate run ids
            rows.append(group.iloc[int(np.argmax(group[LOSS_COLUMN].values))])
            continue
        if len(group) < MIN_RUNS_PER_TOPK * topk:
            continue
        top = group.nlargest(topk, LOSS_COLUMN)
        row = top.iloc[0].copy()
        for col in (LOSS_COLUMN, VAL_LOSS_COLUMN, NLL_COLUMN):
            row[col] = top[col].mean()
        rows.append(row)
    best = pd.DataFrame(rows).sort_values(FLOPS_COLUMN)
    keep, running = [], -np.inf
    for idx, row in best.iterrows():
        if row[LOSS_COLUMN] > running:
            keep.append(row.rename(idx))
            running = row[LOSS_COLUMN]
    return pd.DataFrame(keep)


def load_frontier(which: str, topk: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The compute -> ROC AUC frontier to fit, indexed by source run, plus the run table it
    came from (for the run-level bootstrap).

    "big_prior": max_features 16, num_datapoints_max 400, the data of rounds 1-4. Those are
        the s2 runs, tuned for a 1e14 budget, so below ~1e13 the frontier is a tuning
        artefact (see the round-5 note in the module docstring). With topk = 1 this
        reproduces the notebook's export in `DATA_PATH`.
    "all": every run in the preprocessed table, so each FLOP level is represented by the
        run that was actually tuned for it.
    """
    table = pd.read_csv(ALL_RUNS_PATH).set_index("run")
    table.index = table.index.astype(int)
    if which == "big_prior":
        table = table[(table["config/max_features"] == 16) & (table["config/num_datapoints_max"] == 400)]  # noqa: PLR2004
    elif which != "all":
        raise ValueError(which)
    return frontier_from_table(table, topk), table


def load_seed_noise(flops: np.ndarray, auc: np.ndarray) -> None:
    """Fill NOISE with per-point sigma in ROC AUC units from the seed study's CV table."""
    text = SEED_CV_PATH.read_text().splitlines()
    cv = {}
    for line in text:
        parts = [p.strip() for p in line.split("&")]
        if len(parts) < 6 or not parts[0].replace(".", "").isdigit():  # noqa: PLR2004
            continue
        cv[float(parts[0])] = float(parts[-1].replace("\\\\", "").strip())
    levels = np.array(sorted(cv))
    for f, a in zip(flops, auc):
        nearest = levels[np.argmin(np.abs(np.log10(levels) - np.log10(f)))]
        if f > levels.max():
            nearest = levels.max()
        NOISE["sigma_auc"][float(f)] = cv[nearest] * float(a)
        NOISE["auc"][float(f)] = float(a)


# --- driver -------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, default=None, help="only fit forms introduced up to this round")
    parser.add_argument(
        "--frontier",
        choices=["big_prior", "all"],
        default="big_prior",
        help="which runs the compute -> ROC AUC frontier is built from (default: the rounds-1-4 big-prior subset)",
    )
    parser.add_argument("--topk", type=int, default=1, help="frontier = mean of the k best runs per FLOP level")
    parser.add_argument("--bootstrap", type=int, default=0, help="run-level bootstrap resamples for ceiling CIs")
    parser.add_argument("--seed-weights", action="store_true", help="weight fits by the seed-study noise")
    parser.add_argument("--no-loro", action="store_true", help="skip the leave-one-run-out holdout")
    return parser.parse_args()


def describe_frontier(flops: np.ndarray, auc: np.ndarray, runs: np.ndarray) -> None:
    print(
        f"{len(flops)} Pareto points, FLOPs 1e{np.log10(flops.min()):.1f} .. 1e{np.log10(flops.max()):.1f}, "
        f"ROC AUC {auc.min():.4f} .. {auc.max():.4f}"
    )
    print(
        f"{pd.Series(runs).nunique()} distinct training runs behind those {len(runs)} points; "
        f"largest run contributes {int(pd.Series(runs).value_counts().max())} checkpoints"
    )
    for threshold, info in holdout_contamination(flops, runs).items():
        print(
            f"  cut at {threshold:.2e}: {info['same_run_as_train']}/{info['n_test']} test points "
            f"come from a run already in the training set"
        )


def output_dir(args: argparse.Namespace, round_number: int) -> Path:
    suffix = "" if args.frontier == "big_prior" else f"_{args.frontier}"
    if args.topk > 1:
        suffix += f"_top{args.topk}"
    if args.seed_weights:
        suffix += "_seedw"
    out = FIGURE_ROOT / f"round_{round_number}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    sns.set_style("white")
    sns.set_context("talk")

    df, table = load_frontier(args.frontier, args.topk)
    source = "big prior" if args.frontier == "big_prior" else "all runs"
    POINT_LABEL["text"] = f"Pareto points ({source}" + (f", top-{args.topk} mean)" if args.topk > 1 else ")")
    order = np.argsort(df[FLOPS_COLUMN].values)
    flops = df[FLOPS_COLUMN].values.astype(float)[order]
    auc = df[LOSS_COLUMN].values.astype(float)[order]
    val_loss = df[VAL_LOSS_COLUMN].values.astype(float)[order]
    nll = df[NLL_COLUMN].values.astype(float)[order]
    runs = df.index.values[order]  # source training run; several points share one run
    describe_frontier(flops, auc, runs)

    tabpfn_auc = None
    if TABPFN_BASELINE_PATH.exists():
        tabpfn_auc = float(pd.read_csv(TABPFN_BASELINE_PATH, index_col=0).loc["tabarena", "roc_auc"])

    if args.seed_weights:
        load_seed_noise(flops, auc)
        sig = np.array([NOISE["sigma_auc"][f] for f in flops])
        print(f"seed-noise weights on: sigma(AUC) {sig.min():.4f} .. {sig.max():.4f} across the frontier")

    stage1_params, loss_law = fit_val_loss_law(flops, val_loss)
    print(
        f"stage 1 (validation loss): irreducible loss E_L = {np.exp(stage1_params[0]):.4f}, "
        f"exponent alpha = {-stage1_params[2]:.4f}"
    )

    forms = build_forms(auc_max=float(auc.max()), loss_law=loss_law, flops=flops, val_loss=val_loss, nll=nll)
    round_number = args.round if args.round is not None else max(s.round for s in forms)
    forms = [s for s in forms if s.round <= round_number]
    print(f"round {round_number}: {len(forms)} forms\n")

    for spec in forms:
        fit_form(spec, flops, auc)
        holdout_score(spec, flops, auc)
        group_holdout_score(spec, flops, auc, runs)
        ladder_score(spec, flops, auc)
        if not args.no_loro:
            loro_score(spec, flops, auc, runs)
        profile_ceiling(spec, flops, auc)
    if args.bootstrap:
        bootstrap_runs(forms, table, args.bootstrap, args.topk, np.random.default_rng(BOOTSTRAP_SEED))
    summary = report(forms, tabpfn_auc)

    out = output_dir(args, round_number)
    best = sorted(forms, key=rank_key)[:5]
    figures = {
        "all_forms_per_form": plot_small_multiples(forms, flops, auc, tabpfn_auc),
        "top5_comparison": plot_comparison(best, flops, auc, tabpfn_auc, "Top 5 by leave-future-out ladder"),
        "top5_gap": plot_gap(best, flops, auc, "Gap view: an irreducible gap bends flat"),
        "ceiling_profile": plot_ceiling_profile(forms, float(auc.max())),
        "stage1_val_loss": plot_val_loss_stage(flops, val_loss, stage1_params),
    }
    for name, fig in figures.items():
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=150)
        plt.close(fig)
    summary.to_csv(out / "summary.csv")
    if args.bootstrap:
        boots = {s.name: s.result["bootstrap"]["ceilings"] for s in forms if s.result.get("bootstrap")}
        pd.DataFrame({k: pd.Series(v) for k, v in boots.items()}).to_csv(out / "bootstrap_ceilings.csv", index=False)
    df.to_csv(out / "frontier.csv")
    print(f"\nwrote figures + summary to {out}")


if __name__ == "__main__":
    main()
