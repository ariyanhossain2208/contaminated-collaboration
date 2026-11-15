import argparse

import pandas as pd

from common import GENDERS, LABELS, groups_in, load, show
from occupation_labels import stereotype_label


def is_congruent(gender, label):
    return (gender == "male" and label == "Male-stereotype") or \
           (gender == "female" and label == "Female-stereotype")


def add_congruence(df, threshold=None):
    df = df.copy()
    if threshold:
        df["label"] = [stereotype_label(p, threshold) if pd.notna(p) else "Unmatched"
                       for p in df["pct_women"]]
    df["congruent"] = [is_congruent(g, lab) for g, lab in zip(df["gender"], df["label"])]
    return df


def label_breakdown(df, groups):
    rows = []
    for g in groups:
        for gender in GENDERS:
            sub = df[(df["group"] == g) & (df["gender"] == gender)]
            row = {"group": g, "gender": gender, "n": len(sub)}
            row.update({lab: int((sub["label"] == lab).sum()) for lab in LABELS})
            row["congruent_%"] = 100 * sub["congruent"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def scr_table(df, groups):
    rows = []
    for g in groups:
        sub = df[df["group"] == g]
        rows.append({"group": g, "n": len(sub), "congruent": int(sub["congruent"].sum()),
                     "scr": sub["congruent"].mean()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Stereotype Congruence Rate")
    parser.add_argument("--phase", choices=["a", "b"], default="a")
    parser.add_argument("--threshold", type=float, help="relabel occupations at this %% cutoff")
    args = parser.parse_args()

    df = add_congruence(load(args.phase, scores=False, labels=True), args.threshold)
    groups = groups_in(df)
    cutoff = f" ({args.threshold:g}% threshold)" if args.threshold else ""

    show(label_breakdown(df, groups), f"Occupation labels, phase {args.phase.upper()}{cutoff}", digits=1)
    show(scr_table(df, groups), "Stereotype Congruence Rate")


if __name__ == "__main__":
    main()
