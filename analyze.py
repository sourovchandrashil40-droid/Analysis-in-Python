"""Analyse the checkout A/B test: data checks, primary/secondary/guardrail metrics, segments.

Usage:
    python generate_data.py   # creates data/experiment.csv
    python analyze.py         # prints the report, writes results/summary.txt and charts

Pre-registered plan (fixed BEFORE looking at results, see README):
    primary metric     conversion rate (purchase / user who reached checkout)
    secondary metric   revenue per user
    guardrail          refund rate among buyers may not rise by more than 1.0 pp
    alpha 0.05 (two-sided), power 80%, minimum detectable effect 1.0 pp on an 11% baseline
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "experiment.csv"
RESULTS = ROOT / "results"

ALPHA, POWER = 0.05, 0.80
BASELINE, MDE = 0.11, 0.010
GUARDRAIL_MARGIN = 0.010
N_BOOT, SEED = 2000, 123

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, BASELINE_C = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.family": "sans-serif", "font.size": 10, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left",
})


# ---------- statistics helpers ----------

def two_prop_test(x_c, n_c, x_t, n_t, alpha=ALPHA):
    """Treatment minus control. z-test uses the pooled SE; the CI uses the unpooled SE."""
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    p_pool = (x_c + x_t) / (n_c + n_t)
    z = diff / np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"p_c": p_c, "p_t": p_t, "diff": diff, "rel": diff / p_c, "se": se,
            "lo": diff - zc * se, "hi": diff + zc * se, "z": z,
            "p": 2 * stats.norm.sf(abs(z)), "n_c": n_c, "n_t": n_t}


def _n_float(p_base, mde, alpha=ALPHA, power=POWER):
    p2 = p_base + mde
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z ** 2 * (p_base * (1 - p_base) + p2 * (1 - p2)) / mde ** 2


def n_per_arm(p_base, mde):
    """Users needed in EACH arm to detect an absolute lift of `mde`."""
    return int(np.ceil(_n_float(p_base, mde)))


def mde_for_n(p_base, n):
    """Smallest absolute lift detectable with `n` users per arm at the planned power."""
    return brentq(lambda m: _n_float(p_base, m) - n, 1e-5, 0.2)


def pp(x):
    return f"{x * 100:+.2f} pp"


def fp(p):
    return "< 0.0001" if p < 0.0001 else f"{p:.4f}"


# ---------- analysis ----------

def main():
    RESULTS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    out = []

    def say(line=""):
        out.append(line)
        print(line)

    c, t = df[df.arm == "control"], df[df.arm == "treatment"]
    n_c, n_t = len(c), len(t)

    say("== 1. DATA QUALITY ==")
    say(f"users: {len(df):,} | duplicate user_ids: {df.user_id.duplicated().sum()} | "
        f"missing values: {int(df.isna().sum().sum())}")
    say(f"exposure window: {df.exposure_date.min()} to {df.exposure_date.max()}")

    say("\n== 2. SAMPLE RATIO MISMATCH (expected 50/50) ==")
    chi = stats.chisquare([n_c, n_t])
    say(f"control {n_c:,} | treatment {n_t:,} | chi-square p = {chi.pvalue:.3f} -> "
        f"{'OK, no SRM (threshold p < 0.001)' if chi.pvalue > 0.001 else 'SRM DETECTED, do not trust results'}")

    say("\n== 3. POWER PLAN ==")
    need = n_per_arm(BASELINE, MDE)
    say(f"to detect {pp(MDE)} on a {BASELINE:.0%} baseline (alpha {ALPHA}, power {POWER:.0%}): "
        f"{need:,} users per arm")
    say(f"actual users per arm: {n_c:,} / {n_t:,} -> smallest detectable lift at 80% power: "
        f"{pp(mde_for_n(BASELINE, min(n_c, n_t)))}")

    say("\n== 4. PRIMARY METRIC: CONVERSION ==")
    prim = two_prop_test(c.converted.sum(), n_c, t.converted.sum(), n_t)
    say(f"control {prim['p_c']:.2%} | treatment {prim['p_t']:.2%}")
    say(f"lift {pp(prim['diff'])} ({prim['rel']:+.1%} relative) | 95% CI [{pp(prim['lo'])}, {pp(prim['hi'])}] "
        f"| z = {prim['z']:.2f} | p = {prim['p']:.4f}")

    say("\n== 5. SECONDARY METRIC: REVENUE PER USER (zeros included) ==")
    rng = np.random.default_rng(SEED)
    rv_c, rv_t = c.order_value.values, t.order_value.values
    welch = stats.ttest_ind(rv_t, rv_c, equal_var=False)
    boots = np.array([rng.choice(rv_t, n_t).mean() - rng.choice(rv_c, n_c).mean()
                      for _ in range(N_BOOT)])
    b_lo, b_hi = np.percentile(boots, [2.5, 97.5])
    d_rpu = rv_t.mean() - rv_c.mean()
    say(f"control EUR {rv_c.mean():.2f} | treatment EUR {rv_t.mean():.2f} | difference EUR {d_rpu:+.2f} "
        f"({d_rpu / rv_c.mean():+.1%})")
    say(f"Welch t-test p = {welch.pvalue:.4f} | bootstrap 95% CI [EUR {b_lo:+.2f}, EUR {b_hi:+.2f}] "
        f"({N_BOOT:,} resamples)")
    buy_c, buy_t = c[c.converted == 1].order_value, t[t.converted == 1].order_value
    aov = stats.ttest_ind(buy_t, buy_c, equal_var=False)
    say(f"order value among buyers: control EUR {buy_c.mean():.2f} | treatment EUR {buy_t.mean():.2f} | "
        f"Welch p = {aov.pvalue:.3f} (descriptive only: conditioning on buying can bias a comparison)")

    say("\n== 6. GUARDRAIL: REFUND RATE AMONG BUYERS ==")
    g = two_prop_test(c.refunded.sum(), c.converted.sum(),
                      t.refunded.sum(), t.converted.sum())
    say(f"control {g['p_c']:.2%} | treatment {g['p_t']:.2%} | difference {pp(g['diff'])} | "
        f"95% CI [{pp(g['lo'])}, {pp(g['hi'])}] | p = {g['p']:.3f}")
    say(f"allowed worsening: {pp(GUARDRAIL_MARGIN)} -> upper CI bound {pp(g['hi'])} is "
        f"{'BELOW the margin: guardrail passed' if g['hi'] < GUARDRAIL_MARGIN else 'ABOVE the margin: cannot rule out harm'}")

    say("\n== 7. SEGMENTS BY DEVICE (Bonferroni, 2 segments) ==")
    segs = {"Overall": prim}
    for dev in ["mobile", "desktop"]:
        s = df[df.device == dev]
        sc, st = s[s.arm == "control"], s[s.arm == "treatment"]
        r = two_prop_test(sc.converted.sum(), len(sc), st.converted.sum(), len(st))
        segs[dev.capitalize()] = r
        say(f"{dev:8s} n = {len(sc):,}/{len(st):,} | control {r['p_c']:.2%} | treatment {r['p_t']:.2%} | "
            f"lift {pp(r['diff'])} | 95% CI [{pp(r['lo'])}, {pp(r['hi'])}] | "
            f"p = {fp(r['p'])} (Bonferroni-adjusted {fp(min(1, r['p'] * 2))})")
    m, d = segs["Mobile"], segs["Desktop"]
    z_int = (m["diff"] - d["diff"]) / np.sqrt(m["se"] ** 2 + d["se"] ** 2)
    say(f"difference in lift mobile vs desktop: {pp(m['diff'] - d['diff'])} | z = {z_int:.2f} | "
        f"p = {fp(2 * stats.norm.sf(abs(z_int)))}")

    say("\n== 8. STABILITY OVER TIME (novelty check) ==")
    df["week"] = np.where(pd.to_datetime(df.exposure_date) < pd.Timestamp("2026-09-08"), "week 1", "week 2")
    for wk, s in df.groupby("week"):
        sc, st = s[s.arm == "control"], s[s.arm == "treatment"]
        r = two_prop_test(sc.converted.sum(), len(sc), st.converted.sum(), len(st))
        say(f"{wk}: lift {pp(r['diff'])} | 95% CI [{pp(r['lo'])}, {pp(r['hi'])}]")

    say("\n== 9. DECISION ==")
    ok_primary = prim["p"] < ALPHA and prim["lo"] > 0
    ok_guard = g["hi"] < GUARDRAIL_MARGIN
    say(f"primary metric significant and positive: {ok_primary} | guardrail passed: {ok_guard}")
    say("-> meets the pre-registered ship criteria (check the segment results before rollout)"
        if ok_primary and ok_guard else "-> does NOT meet the ship criteria: extend the test or iterate")

    (RESULTS / "summary.txt").write_text("\n".join(out) + "\n")
    chart_lift(segs)
    chart_bootstrap(boots, b_lo, b_hi)


# ---------- charts ----------

def chart_lift(segs):
    names = list(segs)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ys = np.arange(len(names))
    for y, n in zip(ys, names):
        r = segs[n]
        lo, d, hi = r["lo"] * 100, r["diff"] * 100, r["hi"] * 100
        ax.hlines(y, lo, hi, color=BLUE, linewidth=2)
        ax.plot(d, y, "o", markersize=8, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(hi + 0.15, y, f"{d:+.2f} pp  [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=9,
                fontweight="bold", color=INK)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    lo_all = min(segs[n]["lo"] for n in names) * 100
    hi_all = max(segs[n]["hi"] for n in names) * 100
    ax.set_xlim(min(-0.5, lo_all - 0.3), hi_all + 2.6)
    ax.set_xlabel("Conversion lift, treatment minus control (percentage points)")
    ax.set_title("Conversion lift with 95% confidence interval")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_C)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULTS / "lift_by_segment.png", dpi=150)
    plt.close(fig)


def chart_bootstrap(boots, lo, hi):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(boots, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(0, color=INK2, linewidth=1.2)
    for x in (lo, hi):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
    top = ax.get_ylim()[1]
    ax.text((lo + hi) / 2, top * 1.02, f"95% CI: €{lo:+.2f} to €{hi:+.2f}", ha="center",
            fontsize=9, fontweight="bold", color=INK)
    ax.set_ylim(0, top * 1.12)
    ax.xaxis.set_major_formatter(lambda v, _: f"€{v:.1f}")
    ax.set_xlabel("Revenue per user, treatment minus control (0 = no difference)")
    ax.yaxis.set_visible(False)
    ax.set_title("Bootstrap distribution of the revenue-per-user difference")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE_C)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(RESULTS / "revenue_bootstrap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
