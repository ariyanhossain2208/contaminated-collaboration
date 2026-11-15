
import argparse

import pandas as pd
import spacy
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import GENDERS, groups_in, load, phase_path, show

DEFAULT_MODEL = "checkpoints/agency-bert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
nlp = spacy.load("en_core_web_sm")


def load_model(model_dir=DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE).eval()
    return tokenizer, model


def split_sentences(text):
    return [s.text.strip() for s in nlp(text).sents if s.text.strip()]


@torch.no_grad()
def classify(sentence, tokenizer, model):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    probs = model(**inputs).logits.softmax(dim=-1)[0]
    label = int(probs.argmax())
    return label, float(probs[label])


def agency_score(text, tokenizer, model):
    results = [classify(s, tokenizer, model) for s in split_sentences(text)]
    if not results:
        return 0.0, 0.0
    labels, confs = zip(*results)
    return sum(labels) / len(labels), sum(confs) / len(confs)


def score_essays(essays, model_dir=DEFAULT_MODEL):
    tokenizer, model = load_model(model_dir)
    print(f"scoring {len(essays)} essays on {DEVICE}")
    rows = []
    for essay_id, text in tqdm(zip(essays["id"], essays["essay"]), total=len(essays), unit="essay"):
        per_ac, con_ac = agency_score(text, tokenizer, model)
        rows.append({"id": essay_id, "per_ac": round(per_ac, 6), "con_ac": round(con_ac, 6)})
    return pd.DataFrame(rows)


def gap_table(df, groups):
    rows = []
    for g in groups:
        sub = df[df["group"] == g]
        means = {s: sub.loc[sub["gender"] == s, "per_ac"].mean() for s in GENDERS}
        counts = {s: int((sub["gender"] == s).sum()) for s in GENDERS}
        rows.append({"group": g, "n_male": counts["male"], "n_female": counts["female"],
                     "male": means["male"], "female": means["female"],
                     "gap": means["male"] - means["female"]})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Agency scores and the agentic gap")
    parser.add_argument("--phase", choices=["a", "b"], default="a")
    parser.add_argument("--score", action="store_true",
                        help="run the classifier and overwrite data/phase_<x>/agency_scores.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="classifier checkpoint directory")
    args = parser.parse_args()

    if args.score:
        essays = load(args.phase, scores=False)
        scores = score_essays(essays, args.model)
        path = phase_path(args.phase, "agency_scores.csv")
        scores.to_csv(path, index=False)
        print(f"saved {path}")

    df = load(args.phase)
    show(gap_table(df, groups_in(df)), f"Agentic gap, phase {args.phase.upper()}")


if __name__ == "__main__":
    main()
