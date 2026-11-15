import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

GROUPS = ["control", "neutral", "biased"]
GENDERS = ["male", "female"]
LABELS = ["Male-stereotype", "Female-stereotype", "Neutral"]
PRONOUNS = {
    "male": {"he": "he", "He": "He", "his": "his"},
    "female": {"he": "she", "He": "She", "his": "her"},
}

ESSAY_FILES = {
    "a": DATA / "phase_a" / "essays.csv",
    "b": DATA / "phase_b" / "participants.csv",
}
ID_COLUMNS = {"a": "essay_id", "b": "pid"}


def phase_path(phase, name):
    return DATA / f"phase_{phase}" / name


def read_or_exit(path):
    if not Path(path).exists():
        raise SystemExit(f"{path} not found. Phase B participant data is not distributed "
                         "with this repository (see the README).")
    return pd.read_csv(path)


def load(phase, scores=True, labels=False):
    """Essays for one phase, with agency scores and/or occupation labels joined on."""
    df = read_or_exit(ESSAY_FILES[phase]).rename(columns={ID_COLUMNS[phase]: "id"})
    if scores:
        df = df.merge(read_or_exit(phase_path(phase, "agency_scores.csv")), on="id", how="left")
    if labels:
        lab = read_or_exit(phase_path(phase, "occupation_labels.csv"))
        df = df.merge(lab.drop(columns="occupation"), on="id", how="left")
    return df


def groups_in(df):
    return [g for g in GROUPS if g in set(df["group"])]


def fill_biography(template, name, gender):
    return template.format(name=name, **PRONOUNS[gender])


def groq_client():
    from groq import Groq

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("set the GROQ_API_KEY environment variable first")
    return Groq(api_key=key)


def fmt_p(p):
    if pd.isna(p):
        return ""
    return "<.001" if p < 0.001 else f"{p:.3f}"


def show(df, title=None, digits=3):
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    df = df.copy()
    if "p" in df.columns:
        df["p"] = df["p"].map(fmt_p)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.{digits}f}", na_rep=""))
