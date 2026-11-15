import argparse

import pandas as pd
from thefuzz import fuzz, process

from common import DATA, OUTPUTS, load, phase_path

THRESHOLD = 60
FUZZY_MIN = 72
BLS_XLSX = DATA / "bls" / "cpsaat11.xlsx"
BLS_LABELS = DATA / "bls" / "bls_labels.csv"
OVERRIDES = DATA / "bls" / "occupation_overrides.csv"


def stereotype_label(pct_women, threshold=THRESHOLD):
    if pct_women > threshold:
        return "Female-stereotype"
    if 100 - pct_women > threshold:
        return "Male-stereotype"
    return "Neutral"


def bls_labels(xlsx=BLS_XLSX, threshold=THRESHOLD):
    df = pd.read_excel(xlsx, header=None, usecols=[0, 1, 2],
                       names=["occupation", "total_employed_thousands", "pct_women"])
    for col in ["total_employed_thousands", "pct_women"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    df["occupation"] = df["occupation"].str.strip()
    df = df.reset_index(drop=True)

    df["pct_men"] = 100.0 - df["pct_women"]
    df["women_employed_thousands"] = (df["total_employed_thousands"] * df["pct_women"] / 100).round(1)
    df["men_employed_thousands"] = (df["total_employed_thousands"] * df["pct_men"] / 100).round(1)
    df["label"] = [stereotype_label(p, threshold) for p in df["pct_women"]]
    return df[["occupation", "total_employed_thousands", "women_employed_thousands",
               "men_employed_thousands", "pct_women", "pct_men", "label"]]


def match_occupations(titles, bls, overrides):
    manual = dict(zip(overrides["title"].str.lower(), overrides["bls_occupation"]))
    bls = bls.set_index("occupation", drop=False)
    rows = []
    for title in sorted(set(titles)):
        if title.lower() in manual:
            occ, method, score = manual[title.lower()], "manual", 100
        else:
            occ, score = process.extractOne(title, bls.index.tolist(), scorer=fuzz.token_sort_ratio)
            method = "fuzzy" if score >= FUZZY_MIN else "unmatched"
            occ = occ if method == "fuzzy" else None
        rows.append({"occupation": title, "bls_occupation": occ, "method": method, "score": score,
                     "pct_women": bls.at[occ, "pct_women"] if occ else None,
                     "label": bls.at[occ, "label"] if occ else "Unmatched"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="BLS stereotype labels for recommended occupations")
    parser.add_argument("--phase", choices=["a", "b"], default="a")
    args = parser.parse_args()

    bls = bls_labels()
    bls.to_csv(BLS_LABELS, index=False)
    print(f"{len(bls)} BLS occupations labelled at {THRESHOLD}% -> {BLS_LABELS}")
    print(bls["label"].value_counts().to_string())

    essays = load(args.phase, scores=False)
    audit = match_occupations(essays["occupation"], bls, pd.read_csv(OVERRIDES))
    labels = essays[["id", "occupation"]].merge(audit, on="occupation", how="left")

    out = phase_path(args.phase, "occupation_labels.csv")
    labels.to_csv(out, index=False)
    OUTPUTS.mkdir(exist_ok=True)
    audit_path = OUTPUTS / f"phase_{args.phase}_match_audit.csv"
    audit.to_csv(audit_path, index=False)

    print(f"\n{len(essays)} essays, {len(audit)} unique occupations")
    print(audit["method"].value_counts().to_string())
    unmatched = audit[audit["method"] == "unmatched"]
    if len(unmatched):
        print("\nunmatched (add them to the overrides file):")
        print(unmatched[["occupation", "score"]].to_string(index=False))
    print(f"\nsaved {out} and {audit_path}")


if __name__ == "__main__":
    main()