import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, Trainer,
                          TrainingArguments)

from common import DATA


def main():
    parser = argparse.ArgumentParser(description="Train the agentic/communal classifier")
    parser.add_argument("--data", default=DATA / "lac", help="folder with train/validation/test.csv")
    parser.add_argument("--base", default="bert-base-cased")
    parser.add_argument("--out", default="checkpoints/agency-bert")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    files = {split: str(Path(args.data) / f"{split}.csv") for split in ("train", "validation", "test")}
    dataset = load_dataset("csv", data_files=files)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    dataset = dataset.map(lambda b: tokenizer(b["text"], padding="max_length", truncation=True,
                                              max_length=512), batched=True)

    torch.manual_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2)

    def metrics(pred):
        preds = pred.predictions.argmax(-1)
        return {"accuracy": accuracy_score(pred.label_ids, preds),
                "f1_macro": f1_score(pred.label_ids, preds, average="macro")}

    out = Path(args.out)
    training_args = TrainingArguments(
        output_dir=str(out / "runs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_total_limit=2,
        report_to="none",
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"],
                      eval_dataset=dataset["validation"], compute_metrics=metrics)
    trainer.train()

    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    pred = trainer.predict(dataset["test"])
    report = classification_report(pred.label_ids, np.argmax(pred.predictions, axis=-1),
                                   target_names=["communal", "agentic"], digits=4)
    (out / "test_report.txt").write_text(report)
    print(report)
    print(f"checkpoint saved to {out}")


if __name__ == "__main__":
    main()
