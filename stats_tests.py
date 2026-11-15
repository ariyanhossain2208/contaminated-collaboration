import argparse
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.power import TTestIndPower

from common import GENDERS, groups_in, load, show
from scr import add_congruence

AGENCY_TESTS = ["describe", "gap", "bootstrap", "anova", "condition", "power", "normality"]
SCR_TESTS = ["chi2", "pairwise"]
INTERACTION_TESTS = ["prompts", "likert"]
PAIR_ORDER = ["biased", "control", "neutral"]   
LIKERT = {
    "q2_career_influence": "Q2 AI influence on recommended career",
    "q3_writing_influence": "Q3 AI influence on writing",
    "q7_counterfactual": "Q7 difference without AI",
}


# building blocks 

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled


def cohens_h(p1, p2):
    return abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    half = stats.sem(x) * stats.t.ppf((1 + level) / 2, len(x) - 1)
    return x.mean() - half, x.mean() + half


def welch(a, b, level=0.95):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    diff = a.mean() - b.mean()
    half = np.sqrt(va + vb) * stats.t.ppf((1 + level) / 2, df)
    return {"diff": diff, "t": t, "df": df, "p": p, "d": cohens_d(a, b),
            "ci_low": diff - half, "ci_high": diff + half}


def bootstrap_diff(a, b, n_boot=50_000, seed=42):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def mann_whitney(a, b):
    u, p = stats.mannwhitneyu(a, b)
    return min(u, len(a) * len(b) - u), p


def split_by_gender(df, col="per_ac"):
    return (df.loc[df["gender"] == "male", col].to_numpy(),
            df.loc[df["gender"] == "female", col].to_numpy())


# agency gap 

def describe(df, groups):
    rows = []
    for g in groups:
        for gender in GENDERS:
            x = df.loc[(df["group"] == g) & (df["gender"] == gender), "per_ac"]
            lo, hi = mean_ci(x)
            rows.append({"group": g, "gender": gender, "n": len(x), "mean": x.mean(),
                         "sd": x.std(), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def gap_tests(df, groups):
    rows = []
    for g in groups:
        m, f = split_by_gender(df[df["group"] == g])
        res = welch(m, f)
        rows.append({"group": g, "n_m": len(m), "n_f": len(f), "male": m.mean(),
                     "female": f.mean(), "gap": res.pop("diff"), **res})
    return pd.DataFrame(rows)


def bootstrap_gaps(df, groups, n_boot=50_000, seed=42):
    rows = []
    for g in groups:
        m, f = split_by_gender(df[df["group"] == g])
        lo, hi = bootstrap_diff(m, f, n_boot, seed)
        rows.append({"group": g, "gap": m.mean() - f.mean(), "boot_low": lo, "boot_high": hi})
    return pd.DataFrame(rows)


def two_way_anova(df):
    model = smf.ols("per_ac ~ C(gender) * C(group)", data=df).fit()
    table = anova_lm(model, typ=2)
    ss_res = table.loc["Residual", "sum_sq"]
    table["partial_eta2"] = table["sum_sq"] / (table["sum_sq"] + ss_res)
    table.loc["Residual", "partial_eta2"] = np.nan
    table.index = ["gender", "condition", "gender x condition", "residual"]
    table["df"] = table["df"].astype(int)
    table = table.reset_index().rename(columns={"index": "source", "sum_sq": "ss", "PR(>F)": "p"})
    return table, model.rsquared, model.rsquared_adj


def condition_effects(df, groups):
    omnibus, pairs = [], []
    for gender in GENDERS:
        sub = df[df["gender"] == gender]
        samples = [sub.loc[sub["group"] == g, "per_ac"].to_numpy() for g in groups]
        if len(groups) == 2:
            res = welch(samples[1], samples[0])
            omnibus.append({"gender": gender, "test": f"t ({groups[1]} - {groups[0]})",
                            "stat": res["t"], "df": f"{res['df']:.1f}", "p": res["p"], "d": res["d"]})
            continue
        F, p = stats.f_oneway(*samples)
        n = sum(len(s) for s in samples)
        omnibus.append({"gender": gender, "test": "one-way ANOVA", "stat": F,
                        "df": f"{len(groups) - 1}, {n - len(groups)}", "p": p, "d": np.nan})
        tukey = pairwise_tukeyhsd(sub["per_ac"], sub["group"]).summary().data
        for g1, g2, diff, p_adj, *_ in tukey[1:]:
            pairs.append({"gender": gender, "pair": f"{g1} vs {g2}", "diff": diff, "p": p_adj})
    return pd.DataFrame(omnibus), pd.DataFrame(pairs)


def power_table(df, groups, power=0.8, alpha=0.05):
    solver = TTestIndPower()
    rows = []
    for g in groups:
        m, f = split_by_gender(df[df["group"] == g])
        min_d = solver.solve_power(nobs1=len(m), ratio=len(f) / len(m), alpha=alpha, power=power)
        rows.append({"group": g, "n_m": len(m), "n_f": len(f), "min_d": min_d,
                     "observed_d": abs(cohens_d(m, f))})
    return pd.DataFrame(rows)


def n_per_cell(d, power=0.8, alpha=0.05):
    return int(np.ceil(TTestIndPower().solve_power(effect_size=d, alpha=alpha, power=power)))


def normality(df, groups):
    rows = []
    for g in groups:
        for gender in GENDERS:
            x = df.loc[(df["group"] == g) & (df["gender"] == gender), "per_ac"]
            W, p = stats.shapiro(x)
            rows.append({"group": g, "gender": gender, "n": len(x), "W": W, "p": p})
    return pd.DataFrame(rows)


# stereotype congruence 

def chi_square(df, groups):
    counts = [[df.loc[df["group"] == g, "congruent"].sum(),
               (~df.loc[df["group"] == g, "congruent"]).sum()] for g in groups]
    chi2, p, dof, _ = stats.chi2_contingency(counts)
    return chi2, dof, p


def pairwise_fisher(df, groups):
    ordered = [g for g in PAIR_ORDER if g in groups]
    rows = []
    for a, b in combinations(ordered, 2):
        ca = df.loc[df["group"] == a, "congruent"]
        cb = df.loc[df["group"] == b, "congruent"]
        odds, p = stats.fisher_exact([[ca.sum(), (~ca).sum()], [cb.sum(), (~cb).sum()]])
        rows.append({"comparison": f"{a} vs {b}", "odds_ratio": odds, "p": p,
                     "h": cohens_h(ca.mean(), cb.mean())})
    return pd.DataFrame(rows)


# command line 

def select(df, args):
    if args.writers:
        if args.phase != "b":
            raise SystemExit("--writers only applies to phase B")
        df = df[df["writer_gender"] == args.writers]
    groups = args.groups or groups_in(df)
    return df[df["group"].isin(groups)], groups


def run_agency(args):
    df, groups = select(load(args.phase), args)
    tests = AGENCY_TESTS if "all" in args.tests else args.tests
    print(f"Agency, phase {args.phase.upper()}: {len(df)} essays ({', '.join(groups)})")

    if "describe" in tests:
        show(describe(df, groups), "Agency score by condition and subject gender")
    if "gap" in tests:
        show(gap_tests(df, groups), "Gender gap (male - female), Welch's t-test")
    if "bootstrap" in tests:
        show(bootstrap_gaps(df, groups, args.n_boot), f"Bootstrap 95% CI on the gap ({args.n_boot:,} resamples)")
    if "anova" in tests:
        table, r2, r2_adj = two_way_anova(df)
        show(table, "Two-way ANOVA: agency ~ gender * condition (type II SS)")
        print(f"R^2 = {r2:.3f}, adjusted R^2 = {r2_adj:.3f}")
    if "condition" in tests:
        omnibus, pairs = condition_effects(df, groups)
        show(omnibus, "Effect of condition within each subject gender")
        if len(pairs):
            show(pairs, "Tukey HSD")
    if "power" in tests:
        show(power_table(df, groups), "Post-hoc power (alpha = .05, power = .80)")
        for d in (0.5, 0.65, 0.8):
            print(f"essays per gender per condition needed for d = {d}: {n_per_cell(d)}")
    if "normality" in tests:
        show(normality(df, groups), "Shapiro-Wilk normality test per cell")


def run_scr(args):
    df = add_congruence(load(args.phase, scores=False, labels=True), args.threshold)
    df, groups = select(df, args)
    tests = SCR_TESTS if "all" in args.tests else args.tests
    cutoff = f", {args.threshold:g}% threshold" if args.threshold else ""
    print(f"SCR, phase {args.phase.upper()}: {len(df)} essays ({', '.join(groups)}{cutoff})")

    if "chi2" in tests:
        if len(groups) < 3:
            print("\n(chi-square skipped: fewer than three conditions, see pairwise)")
        else:
            chi2, dof, p = chi_square(df, groups)
            print(f"\nchi2({dof}) = {chi2:.2f}, p = {p:.3f}")
    if "pairwise" in tests:
        show(pairwise_fisher(df, groups), "Pairwise Fisher's exact tests")


def run_interaction(args):
    df = load("b", scores=False)
    biased, neutral = df[df["group"] == "biased"], df[df["group"] == "neutral"]
    tests = INTERACTION_TESTS if "all" in args.tests else args.tests

    if "prompts" in tests:
        n = pd.concat([biased, neutral])["n_prompts"]
        u, p = mann_whitney(biased["n_prompts"], neutral["n_prompts"])
        print(f"\nPrompts per participant ({len(n)} assistant users): median {n.median():g}, "
              f"IQR {n.quantile(.25):g}-{n.quantile(.75):g}, range {n.min():g}-{n.max():g}")
        print(f"biased vs neutral: U = {u:.1f}, p = {p:.3f}")
    if "likert" in tests:
        rows = []
        for col, item in LIKERT.items():
            b, nt = biased[col].dropna(), neutral[col].dropna()
            u, p = mann_whitney(b, nt)
            rows.append({"item": item, "biased": f"{b.mean():.2f} ± {b.std():.2f}",
                         "neutral": f"{nt.mean():.2f} ± {nt.std():.2f}", "U": u, "p": p})
        show(pd.DataFrame(rows), "Perceived influence (5-point scales, mean ± SD)", digits=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="metric", required=True)

    def add_common(p):
        p.add_argument("--phase", choices=["a", "b"], default="a",
                       help="a = LLM-generated essays, b = participant essays")
        p.add_argument("--groups", nargs="+", help="conditions to include (default: all present)")
        p.add_argument("--writers", choices=["male", "female"],
                       help="phase B only: keep essays by participants of this gender")

    p = sub.add_parser("agency", help="tests on the agentic gap")
    add_common(p)
    p.add_argument("--tests", nargs="+", choices=AGENCY_TESTS + ["all"], default=["describe", "gap", "anova"])
    p.add_argument("--n-boot", type=int, default=50_000)

    p = sub.add_parser("scr", help="tests on stereotype congruence")
    add_common(p)
    p.add_argument("--tests", nargs="+", choices=SCR_TESTS + ["all"], default=["all"])
    p.add_argument("--threshold", type=float, help="relabel occupations at this %% cutoff")

    p = sub.add_parser("interaction", help="phase B assistant use and survey items")
    p.add_argument("--tests", nargs="+", choices=INTERACTION_TESTS + ["all"], default=["all"])

    args = parser.parse_args()
    if args.metric == "agency":
        run_agency(args)
    elif args.metric == "scr":
        run_scr(args)
    else:
        run_interaction(args)


if __name__ == "__main__":
    main()
