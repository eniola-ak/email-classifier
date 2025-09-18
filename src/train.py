# === Baseline: TF-IDF + Logistic Regression ===

import json
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import LABEL2ID, ID2LABEL
from src.preprocess import clean_email
from sklearn.utils.multiclass import unique_labels
import joblib

def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def load_xy(split):
    items = load_jsonl(f"data/processed/{split}.jsonl")
    X = [clean_email(x["text"]) for x in items]
    y = np.array([LABEL2ID[x["label"]] for x in items])
    return X, y

def train_baseline():
    print("\n[+] Loading training and validation data...")
    Xtr, ytr = load_xy("train")
    Xva, yva = load_xy("valid")

    print("[+] Training baseline TF-IDF + Logistic Regression model...")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xva)

    # Dynamically detect present labels
    labels_present = unique_labels(yva, pred)
    target_names = [ID2LABEL[i] for i in labels_present]

    print("\n--- Baseline Model Evaluation ---")
    print(classification_report(
        yva,
        pred,
        labels=labels_present,
        target_names=target_names,
        digits=3
    ))

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, "models/baseline_logreg.joblib")
    print("\n[✓] Baseline model saved to models/baseline_logreg.joblib")

# === Transformer: Fine-tune DistilBERT ===

def train_transformer():
    try:
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer
        )
        import evaluate
        from src.config import MODEL_NAME, MAX_LEN
        from src.dataset import load_hf_dataset
    except ImportError as e:
        raise ImportError(f"Missing package: {e}. Run `pip install torch transformers evaluate`.")

    def tokenize(batch, tok):
        return tok(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )

    # Load & preprocess datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = load_hf_dataset("train") \
        .map(lambda b: tokenize(b, tokenizer), batched=True) \
        .remove_columns(["text"]) \
        .with_format("torch")

    valid_ds = load_hf_dataset("valid") \
        .map(lambda b: tokenize(b, tokenizer), batched=True) \
        .remove_columns(["text"]) \
        .with_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": f1.compute(predictions=preds, references=labels, average="macro")["f1"],
            "f1_micro": f1.compute(predictions=preds, references=labels, average="micro")["f1"]
        }

    training_args = TrainingArguments(
        output_dir="models/intent_distilbert",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    model_dir = "models/intent_distilbert/best"
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"\nSaved fine-tuned transformer model to: {model_dir}")


# === Entry Point ===

if __name__ == "__main__":
    # Uncomment ONE of these to run
    # train_baseline()
    train_transformer()
