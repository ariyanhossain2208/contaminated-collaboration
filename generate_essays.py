import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from groq import Groq

from common import DATA, fill_biography

SECTIONS = ["career_plan", "personal_qualities", "work_environment", "challenge"]
FIELDS = ["essay_id", "group", "bio_id", "gender", "name", "biography", *SECTIONS,
          "occupation_raw", "essay", "word_count", "json_ok", "raw_response",
          "prompt_tokens", "completion_tokens", "finish_reason", "model", "temperature", "timestamp"]


def parse_response(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}

    sections = {key: str(data.get(key) or "").strip() for key in SECTIONS}
    essay = "\n\n".join(v for v in sections.values() if v) or text.strip()
    occupation = str(data.get("occupation") or "").strip()
    return sections, occupation, essay, all(sections.values())


def main():
    parser = argparse.ArgumentParser(description="Generate the Phase A essays")
    parser.add_argument("--out", default=DATA / "phase_a" / "essays_generated.csv")
    parser.add_argument("--bios", default=DATA / "stimuli" / "biographies_phase_a.csv")
    parser.add_argument("--groups", nargs="+", default=["neutral", "biased"], choices=["neutral", "biased"])
    parser.add_argument("--names", type=int, default=100, help="names per gender")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    prompts = DATA / "prompts"
    system = {g: (prompts / f"system_{g}.txt").read_text().strip() for g in args.groups}
    task = (prompts / "task.txt").read_text().strip()
    bios = pd.read_csv(args.bios)
    names = pd.read_csv(DATA / "stimuli" / "names.csv")

    out = Path(args.out)
    done = set(pd.read_csv(out)["essay_id"]) if out.exists() else set()
    client = Groq(max_retries=5)   

    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not done:
            writer.writeheader()

        for group in args.groups:
            for bio in bios.itertuples():
                for gender in ("male", "female"):
                    for name in names.loc[names["gender"] == gender, "name"].head(args.names):
                        essay_id = f"{group}_{bio.bio_id}_{gender}_{name}"
                        if essay_id in done:
                            continue

                        biography = fill_biography(bio.template, name, gender)
                        messages = [{"role": "system", "content": system[group]},
                                    {"role": "user", "content": task.replace("{biography}", biography)}]
                        resp = client.chat.completions.create(model=args.model, messages=messages,
                                                              temperature=args.temperature)
                        text = resp.choices[0].message.content
                        sections, occupation, essay, ok = parse_response(text)

                        writer.writerow({
                            "essay_id": essay_id, "group": group, "bio_id": bio.bio_id,
                            "gender": gender, "name": name, "biography": biography, **sections,
                            "occupation_raw": occupation, "essay": essay,
                            "word_count": len(essay.split()), "json_ok": ok, "raw_response": text,
                            "prompt_tokens": resp.usage.prompt_tokens,
                            "completion_tokens": resp.usage.completion_tokens,
                            "finish_reason": resp.choices[0].finish_reason, "model": args.model,
                            "temperature": args.temperature, "timestamp": datetime.now().isoformat(),
                        })
                        f.flush()
                        print(f"{essay_id}: {occupation or '(no occupation parsed)'}")


if __name__ == "__main__":
    main()