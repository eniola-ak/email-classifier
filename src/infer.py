import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocess import clean_email
from src.config import ID2LABEL

MODEL_DIR = "models/intent_distilbert/best"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

@torch.inference_mode()
def predict_intent(text: str):
    enc = tok(clean_email(text), truncation=True, padding=True, max_length=384, return_tensors="pt").to(DEVICE)
    probs = model(**enc).logits.softmax(-1).squeeze().cpu().numpy()
    pred = int(probs.argmax())
    return {"label": ID2LABEL[pred], "confidence": float(probs[pred]),
            "probs": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}}

if __name__ == "__main__":
    print(predict_intent("I was double charged, please refund."))
