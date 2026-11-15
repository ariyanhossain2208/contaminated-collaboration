"""
    check = classify the body sentences of every biography, both versions
    balance = rewrite body sentences with an LLM until each draft has a 2/2 split

    python biographies.py check
    python biographies.py balance --drafts my_drafts.csv --out outputs/balanced.csv
"""

import argparse

import pandas as pd

from common import DATA, OUTPUTS, fill_biography, groq_client
from agency_gap import DEFAULT_MODEL, classify, load_model, split_sentences

LABEL_NAMES = {0: "communal", 1: "agentic"}
TARGET = 2 # agentic (and communal) sentences wanted among the four body sentences
REWRITE_MODEL = "llama-3.3-70b-versatile"


def check(args, tokenizer, model):
    bios = pd.read_csv(args.bios)
    rows = []
    for bio in bios.itertuples():
        for gender, name in (("male", bio.male_name), ("female", bio.female_name)):
            sentences = split_sentences(fill_biography(bio.template, name, gender))
            if len(sentences) != 6:
                print(f"{bio.bio_id} ({gender}): expected 6 sentences, found {len(sentences)}")
            for i, sent in enumerate(sentences, start=1):
                label, conf = classify(sent, tokenizer, model)
                rows.append({"bio_id": bio.bio_id, "gender": gender, "sentence": i,
                             "body": 2 <= i <= 5, "label": LABEL_NAMES[label],
                             "confidence": round(conf, 4), "text": sent})

    df = pd.DataFrame(rows)
    body = df[df["body"]]
    summary = (body.groupby(["bio_id", "gender"])["label"].value_counts()
               .unstack(fill_value=0).reindex(columns=["agentic", "communal"], fill_value=0))
    summary["balanced"] = (summary["agentic"] == TARGET) & (summary["communal"] == TARGET)
    print(summary.to_string())

    OUTPUTS.mkdir(exist_ok=True)
    df.to_csv(OUTPUTS / "biography_check.csv", index=False)
    print(f"\nsentence-level labels saved to {OUTPUTS / 'biography_check.csv'}")


def rewrite(client, sentence, target):
    style = {
        "agentic": "independent, assertive, achievement-oriented, leadership-focused",
        "communal": "cooperative, warm, supportive, relationship-oriented",
    }[target]
    prompt = (
        f"Rewrite the following sentence so it clearly expresses a {target} trait ({style}). "
        "Keep the same general life event, keep it concise and use simple words. "
        "Leave the placeholders {name}, {he}, {He} and {his} exactly as they are. "
        "Return only the rewritten sentence.\n\n"
        f"Original: {sentence}"
    )
    resp = client.chat.completions.create(model=REWRITE_MODEL, temperature=0.7,
                                          messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content.strip()


def balance(args, tokenizer, model):
    drafts = pd.read_csv(args.drafts)
    client = groq_client()
    results = []

    for bio in drafts.itertuples():
        sentences = split_sentences(bio.template)
        if len(sentences) != 6:
            print(f"{bio.bio_id}: expected 6 sentences, found {len(sentences)}; skipped")
            continue
        print(f"\n{bio.bio_id}")
        body = sentences[1:5]

        for step in range(args.max_iter):
            preds = [classify(s, tokenizer, model) for s in body]
            n_agentic = sum(label for label, _ in preds)
            print(f"  round {step + 1}: {n_agentic} agentic, {4 - n_agentic} communal")
            if n_agentic == TARGET:
                break
            majority = 1 if n_agentic > TARGET else 0
            target = LABEL_NAMES[1 - majority]
            candidates = sorted((conf, i) for i, (label, conf) in enumerate(preds) if label == majority)
            for _, i in candidates[:abs(n_agentic - TARGET)]:
                new = rewrite(client, body[i], target)
                print(f"    {body[i]}\n    -> {new}")
                body[i] = new
        else:
            print("  not balanced after max round")

        labels = [LABEL_NAMES[classify(s, tokenizer, model)[0]] for s in body]
        results.append({"bio_id": bio.bio_id,
                        "template": " ".join([sentences[0], *body, sentences[5]]),
                        "body_labels": ", ".join(labels)})

    OUTPUTS.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nsaved {args.out}")


def main():
    parser = argparse.ArgumentParser(description="Biography stimuli")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="agency classifier checkpoint")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="verify the 2 agentic / 2 communal split")
    p.add_argument("--bios", default=DATA / "stimuli" / "biographies.csv")

    p = sub.add_parser("balance", help="rewrite draft biographies until balanced")
    p.add_argument("--drafts", required=True, help="CSV with bio_id and template columns")
    p.add_argument("--out", default=OUTPUTS / "balanced_biographies.csv")
    p.add_argument("--max-iter", type=int, default=5)

    args = parser.parse_args()
    tokenizer, model = load_model(args.model)
    if args.command == "check":
        check(args, tokenizer, model)
    else:
        balance(args, tokenizer, model)


if __name__ == "__main__":
    main()
