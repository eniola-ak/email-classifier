import joblib
import os
from src.preprocess import clean_email
from src.config import ID2LABEL

model_path = os.path.join("models", "baseline_logreg.joblib")
model = joblib.load(model_path)

def classify_emails(email_texts):
    cleaned = [clean_email(text) for text in email_texts]
    preds = model.predict(cleaned)
    return [{"text": t, "label": ID2LABEL[p]} for t, p in zip(email_texts, preds)]