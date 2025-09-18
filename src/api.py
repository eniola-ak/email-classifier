from fastapi import FastAPI
from pydantic import BaseModel
from src.infer import predict_intent

app = FastAPI(title="Email Intent Classifier")

class EmailIn(BaseModel):
    text: str

@app.post("/classify")
def classify(email: EmailIn):
    return predict_intent(email.text)
