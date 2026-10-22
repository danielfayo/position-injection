"""Pre-registered analysis (written in step 8, run in step 12).

RQ1, for each model: what shape does attack success take across the seven
positions? Every test is a linear probability model (ordinary regression on
0/1 success) with wording as a factor and standard errors clustered by document
(the same 30 documents recur in every cell). Slopes are in proportion units:
0.20 means a 20-point change across the whole document. Logistic regression is
not used here because it breaks down when a position has 0% or 100% success.
Position is coded x = -0.5 (start) to +0.5 (end).

  - Omnibus: do the seven positions differ at all? (joint Wald test)
  - U-shape: two-lines test. A continuous two-segment line is fitted with its
    break at the interior position (2-6) with the lowest success rate. U-shaped
    if the first slope is significantly negative AND the second significantly
    positive. Inverted U: break at the highest interior position, signs reversed.
  - Rising / falling: slope of a straight-line fit.
  - Flat: every position's success rate lies within ±margin of the model's mean,
    using 90% intervals from a bootstrap over documents (an equivalence test).

  Verdict, checked in this order: U-shaped, inverted U, rising, falling, flat,
  otherwise inconclusive. A model's wordings with under 1% or over 99% success
  carry no positional information and are excluded for that model.

RQ2: do the positional effects differ between models? Pooled model with
model-specific linear and quadratic position terms; joint Wald test.

RQ3, Qwen only (logistic regression): does attention to the injection (log,
standardised) predict success? How do position effects change once attention is included? What shape
does attention itself take across positions (same tests, linear regression)?

Robustness: (1) responses listed in analysis/refusals_verified.csv (checked by
hand) are recoded as failures and RQ1 is rerun; (2) RQ1 per wording.

Usage:
    uv run python -m src.analysis                 # main run
    uv run python -m src.analysis --trials trials_pilot_c.jsonl   # any scored trial file
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.05
N_POSITIONS = 7
DEGENERATE = 0.01
BINOMIAL = sm.families.Binomial()
GAUSSIAN = sm.families.Gaussian()


# ---------------------------------------------------------------- helpers

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["x"] = (df["position"] - 1) / (N_POSITIONS - 1) - 0.5
    df["x2"] = df["x"] ** 2
    df["doc_code"] = pd.factorize(df["doc_id"])[0]
    return df


def fit(formula: str, df: pd.DataFrame, family):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.glm(formula, data=df, family=family).fit(
            cov_type="cluster", cov_kwds={"groups": df["doc_code"].to_numpy()})


def wording_term(df: pd.DataFrame) -> str:
    return " + C(wording)" if df["wording"].nunique() > 1 else ""


def coef(result, name: str) -> tuple[float, float]:
    return float(result.params[name]), float(result.pvalues[name])


def joint_p(result, names: list[str]) -> float:
    index = list(result.params.index)
    restriction = np.zeros((len(names), len(index)))
    for row, name in enumerate(names):
        restriction[row, index.index(name)] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(result.wald_test(restriction, scalar=True).pvalue)


def significant(slope: float, p: float, sign: int) -> bool:
    return bool(np.isfinite(p) and p < ALPHA and np.sign(slope) == sign)


def drop_degenerate_wordings(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rates = df.groupby("wording")["success"].mean()
    excluded = sorted(rates[(rates < DEGENERATE) | (rates > 1 - DEGENERATE)].index)
    return df[~df["wording"].isin(excluded)], excluded


# ---------------------------------------------------------------- RQ1 tests

def two_lines(df: pd.DataFrame, breakpoint: int, outcome: str, family) -> dict:
    b = (breakpoint - 1) / (N_POSITIONS - 1) - 0.5
    d = df.assign(xlow=np.minimum(df["x"] - b, 0.0), xhigh=np.maximum(df["x"] - b, 0.0))
    result = fit(f"{outcome} ~ xlow + xhigh" + wording_term(d), d, family)
    slope_low, p_low = coef(result, "xlow")
    slope_high, p_high = coef(result, "xhigh")
    return {"breakpoint": int(breakpoint), "slope_low": slope_low, "p_low": p_low,
            "slope_high": slope_high, "p_high": p_high}


def shape_tests(df: pd.DataFrame, outcome: str = "success", family=GAUSSIAN) -> dict:
    rates = df.groupby("position")[outcome].mean()
    interior = rates.loc[2:N_POSITIONS - 1]
    linear = fit(f"{outcome} ~ x" + wording_term(df), df, family)
    quadratic = fit(f"{outcome} ~ x + x2" + wording_term(df), df, family)
    categorical = fit(f"{outcome} ~ C(position)" + wording_term(df), df, family)
    slope, slope_p = coef(linear, "x")
    u = two_lines(df, int(interior.idxmin()), outcome, family)
    inverted = two_lines(df, int(interior.idxmax()), outcome, family)
    return {
        "omnibus_p": joint_p(categorical, [n for n in categorical.params.index if n.startswith("C(position)")]),
        "linear_slope": slope, "linear_p": slope_p,
        "quadratic": {"x": coef(quadratic, "x"), "x2": coef(quadratic, "x2")},
        "two_lines_u": u, "two_lines_inverted": inverted,
        "is_u": significant(u["slope_low"], u["p_low"], -1) and significant(u["slope_high"], u["p_high"], 1),
        "is_inverted_u": (significant(inverted["slope_low"], inverted["p_low"], 1)
                          and significant(inverted["slope_high"], inverted["p_high"], -1)),
    }


def bootstrap_rates(df: pd.DataFrame, outcome: str, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Success rate at each position in n_boot resamples of the documents."""
    docs, positions = df["doc_code"].to_numpy(), df["position"].to_numpy() - 1
    n_docs = docs.max() + 1
    successes, counts = np.zeros((n_docs, N_POSITIONS)), np.zeros((n_docs, N_POSITIONS))
    np.add.at(successes, (docs, positions), df[outcome].to_numpy(dtype=float))
    np.add.at(counts, (docs, positions), 1)
    draws = rng.integers(0, n_docs, size=(n_boot, n_docs))
    with np.errstate(invalid="ignore", divide="ignore"):   # sparse data only (e.g. pilots)
        return successes[draws].sum(axis=1) / counts[draws].sum(axis=1)


def equivalence(boot: np.ndarray, margin: float) -> dict:
    deviations = boot - boot.mean(axis=1, keepdims=True)
    low, high = np.nanpercentile(deviations, [5, 95], axis=0)
    return {"margin": margin, "deviation_ci90": [[round(a, 4), round(b, 4)] for a, b in zip(low, high)],
            "equivalent": bool(np.all(low > -margin) and np.all(high < margin))}


def verdict(tests: dict, equivalent: bool | None) -> str:
    if tests["is_u"] and not tests["is_inverted_u"]:
        return "U-shaped"
    if tests["is_inverted_u"] and not tests["is_u"]:
        return "inverted U"
    if significant(tests["linear_slope"], tests["linear_p"], 1):
        return "rising"
    if significant(tests["linear_slope"], tests["linear_p"], -1):
        return "falling"
    if equivalent:
        return "flat"
    return "inconclusive"


def analyse_model(df: pd.DataFrame, margin: float, n_boot: int, rng: np.random.Generator) -> dict:
    """RQ1 for one model's trials."""
    df, excluded = drop_degenerate_wordings(df)
    if df.empty:
        return {"verdict": "not analysable", "excluded_wordings": excluded, "n_trials": 0}
    tests = shape_tests(df)
    boot = bootstrap_rates(df, "success", n_boot, rng)
    eq = equivalence(boot, margin)
    rates = df.groupby("position")["success"].mean()
    ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {
        "verdict": verdict(tests, eq["equivalent"]),
        "n_trials": int(len(df)), "excluded_wordings": excluded,
        "rates": [round(float(r), 4) for r in rates], "ci95": [[round(a, 4), round(b, 4)] for a, b in zip(ci_low, ci_high)],
        "tests": tests, "equivalence": eq,
    }


# ---------------------------------------------------------------- RQ2 and RQ3

def compare_models(df: pd.DataFrame) -> dict:
    parts = [drop_degenerate_wordings(group)[0] for _, group in df.groupby("model")]
    pooled = pd.concat(parts)
    pooled = pooled.assign(cell=pooled["model"] + "|" + pooled["wording"])
    result = fit("success ~ C(cell) + x + x2 + C(model):x + C(model):x2", pooled, GAUSSIAN)
    terms = [n for n in result.params.index if n.startswith("C(model)[") and (n.endswith(":x") or n.endswith(":x2"))]
    return {"n_models": int(pooled["model"].nunique()), "interaction_p": joint_p(result, terms),
            "differ": joint_p(result, terms) < ALPHA}


def analyse_attention(df: pd.DataFrame, attention: pd.DataFrame) -> dict:
    df, excluded = drop_degenerate_wordings(df.merge(attention, on="trial_id", how="inner"))
    df = df.assign(log_attn=np.log(df["attn_injection"]), log_task=np.log(df["attn_task"]))
    df = df.assign(z=(df["log_attn"] - df["log_attn"].mean()) / df["log_attn"].std(),
                   z_task=(df["log_task"] - df["log_task"].mean()) / df["log_task"].std())
    w = wording_term(df)
    h3 = fit("success ~ z" + w, df, BINOMIAL)
    without = fit("success ~ x + x2" + w, df, BINOMIAL)
    with_attention = fit("success ~ x + x2 + z" + w, df, BINOMIAL)
    task = fit("success ~ z_task" + w, df, BINOMIAL)
    shape = shape_tests(df, outcome="log_attn", family=GAUSSIAN)
    beta, p = coef(h3, "z")
    return {
        "n_trials": int(len(df)), "excluded_wordings": excluded,
        "h3_attention_predicts_success": {"log_odds_per_sd": beta, "odds_ratio_per_sd": float(np.exp(beta)),
                                          "p": p, "supported": significant(beta, p, 1)},
        "position_terms_without_attention": {"x": coef(without, "x"), "x2": coef(without, "x2")},
        "position_terms_with_attention": {"x": coef(with_attention, "x"), "x2": coef(with_attention, "x2")},
        "exploratory_task_attention": {"log_odds_per_sd": coef(task, "z_task")[0], "p": coef(task, "z_task")[1]},
        "attention_shape": {"verdict": verdict(shape, None).replace("inconclusive", "no clear pattern"),
                            "mean_log_attn_by_position": [round(float(v), 4) for v in
                                                          df.groupby("position")["log_attn"].mean()],
                            "tests": shape},
        "attention_by_position": [round(float(v), 5) for v in df.groupby("position")["attn_injection"].mean()],
    }


# ---------------------------------------------------------------- outputs

def plot_success(models: dict, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    positions = np.arange(1, N_POSITIONS + 1)
    for name, result in models.items():
        if "rates" not in result:
            continue
        rates = np.array(result["rates"]) * 100
        ci = np.array(result["ci95"]) * 100
        line, = ax.plot(positions, rates, marker="o", label=f"{name.split('/')[-1]} ({result['verdict']})")
        ax.fill_between(positions, ci[:, 0], ci[:, 1], color=line.get_color(), alpha=0.15)
    ax.set_xticks(positions, ["1\nstart", "2", "3", "4\nmiddle", "5", "6", "7\nend"])
    ax.set_xlabel("Position of the injected instruction")
    ax.set_ylabel("Attack success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_attention(result: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(1, N_POSITIONS + 1)
    ax.plot(positions, np.array(result["attention_by_position"]) * 100, marker="o", color="tab:purple")
    ax.set_xticks(positions, ["1\nstart", "2", "3", "4\nmiddle", "5", "6", "7\nend"])
    ax.set_xlabel("Position of the injected instruction")
    ax.set_ylabel("Attention on the injection (%)")
    ax.set_title(f"Qwen attention to the injection ({result['attention_shape']['verdict']})")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summary_text(results: dict) -> str:
    lines = [f"# Analysis summary: {results['trials_file']}", ""]
    lines.append("## RQ1: shape of the positional effect, per model")
    for name, r in results["rq1"].items():
        if "tests" not in r:
            lines.append(f"- **{name}**: {r['verdict']}")
            continue
        t = r["tests"]
        rates = ", ".join(f"{v * 100:.0f}" for v in r["rates"])
        lines.append(f"- **{name}**: {r['verdict']}. Success by position (%): {rates}. "
                     f"Omnibus p = {t['omnibus_p']:.3g}; straight-line change start to end {t['linear_slope'] * 100:+.1f} points (p = {t['linear_p']:.3g}); "
                     f"two-lines U test: slopes {t['two_lines_u']['slope_low'] * 100:+.1f} / {t['two_lines_u']['slope_high'] * 100:+.1f} points per document length "
                     f"(p = {t['two_lines_u']['p_low']:.3g} / {t['two_lines_u']['p_high']:.3g}, break at {t['two_lines_u']['breakpoint']}); "
                     f"within ±{r['equivalence']['margin'] * 100:.0f} points everywhere: {r['equivalence']['equivalent']}. "
                     f"Excluded wordings: {r['excluded_wordings'] or 'none'}; n = {r['n_trials']}.")
    if "rq2" in results:
        r = results["rq2"]
        lines += ["", "## RQ2: do the models differ?",
                  f"- Joint test of model × position terms: p = {r['interaction_p']:.3g} ({'differ' if r['differ'] else 'no evidence of a difference'})."]
    if "rq3" in results:
        r = results["rq3"]
        h = r["h3_attention_predicts_success"]
        lines += ["", "## RQ3: attention (Qwen)",
                  f"- Attention predicts success: odds ratio {h['odds_ratio_per_sd']:.2f} per SD of log attention (p = {h['p']:.3g}); H3 {'supported' if h['supported'] else 'not supported'}.",
                  f"- Shape of attention across positions: {r['attention_shape']['verdict']}.",
                  f"- Position terms without / with attention: x {r['position_terms_without_attention']['x'][0]:+.2f} / {r['position_terms_with_attention']['x'][0]:+.2f}, "
                  f"x² {r['position_terms_without_attention']['x2'][0]:+.2f} / {r['position_terms_with_attention']['x2'][0]:+.2f}.",
                  f"- Exploratory: attention to the user's task, odds ratio {np.exp(r['exploratory_task_attention']['log_odds_per_sd']):.2f} per SD (p = {r['exploratory_task_attention']['p']:.3g})."]
    for key, label in [("robustness_refusals", "Robustness: verified refusals recoded as failures"),
                       ("robustness_by_wording", "Robustness: verdict per wording")]:
        if key in results:
            lines += ["", f"## {label}"]
            value = results[key]
            if isinstance(value, str):
                lines.append(f"- {value}")
            else:
                for name, r in value.items():
                    lines.append(f"- {name}: " + (r if isinstance(r, str) else
                                                  ", ".join(f"{k}: {v}" for k, v in r.items())))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pre-registered analysis.")
    parser.add_argument("--trials", default="trials.jsonl")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    margin = config["analysis"]["equivalence_margin"]
    stem = Path(args.trials).stem
    rng = np.random.default_rng(config["seed"])
    scored = pd.read_csv(ROOT / "results" / "scored" / f"{stem}.csv")
    scored = prepare(scored[scored["responded"]].astype({"success": float}))
    out_dir = ROOT / "analysis" / "output" / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"trials_file": args.trials, "equivalence_margin": margin, "alpha": ALPHA,
               "rq1": {name: analyse_model(group, margin, args.n_boot, rng)
                       for name, group in scored.groupby("model")}}
    if scored["model"].nunique() > 1:
        results["rq2"] = compare_models(scored)

    attention_path = ROOT / "results" / "attention" / f"{stem}.jsonl"
    if attention_path.exists():
        attention = pd.read_json(attention_path, lines=True)[["trial_id", "attn_injection", "attn_task"]]
        qwen = scored[scored["model"] == config["attention"]["model"]]
        results["rq3"] = analyse_attention(qwen, attention)
        plot_attention(results["rq3"], out_dir / "attention_by_position.png")

    refusals_path = ROOT / "analysis" / "refusals_verified.csv"
    if refusals_path.exists():
        refused = set(pd.read_csv(refusals_path)["trial_id"])
        recoded = scored.assign(success=np.where(scored["trial_id"].isin(refused), 0.0, scored["success"]))
        results["robustness_refusals"] = {
            name: {"verdict": analyse_model(group, margin, args.n_boot, rng)["verdict"],
                   "recoded": int(group["trial_id"].isin(refused).sum())}
            for name, group in recoded.groupby("model")}
    else:
        results["robustness_refusals"] = "analysis/refusals_verified.csv not found (hand-check flagged responses first)."
    results["robustness_by_wording"] = {
        name: {wording: analyse_model(part, margin, args.n_boot, rng)["verdict"]
               for wording, part in group.groupby("wording")}
        for name, group in scored.groupby("model")}

    plot_success(results["rq1"], out_dir / "success_by_position.png", "Attack success by injection position")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (out_dir / "summary.md").write_text(summary_text(results))
    print(summary_text(results))
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
