import base64
from email import message_from_bytes
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from src.config import ID2LABEL, MAX_LEN

# === Constants ===
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL_DIR = "models/intent_distilbert/best"
NUM_EMAILS = 5


def get_gmail_service():
    creds = None

    # Load saved token
    if Path("token.json").exists():
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If token doesn't exist or expired
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("client_secret.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token
        with open("token.json", "w") as token_file:
            token_file.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def fetch_emails(service, max_results=NUM_EMAILS):
    messages = service.users().messages().list(userId="me", maxResults=max_results).execute().get("messages", [])
    emails = []

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        payload = msg_data.get("payload", {})
        parts = payload.get("parts", [])
        body = ""

        # Try to get plain text body
        for part in parts:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    decoded_bytes = base64.urlsafe_b64decode(data.encode("UTF-8"))
                    body = message_from_bytes(decoded_bytes).get_payload()
                    break

        # Fallback: snippet
        if not body:
            body = msg_data.get("snippet", "")

        emails.append(body)

    return emails


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def classify_emails(emails, tokenizer, model):
    predictions = []

    for text in emails:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN)
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = torch.argmax(outputs.logits, dim=1).item()
            label = ID2LABEL[pred_id]
            predictions.append((label, text[:80]))

    return predictions


if __name__ == "__main__":
    print("[+] Authenticating with Gmail...")
    service = get_gmail_service()

    print(f"[+] Fetching last {NUM_EMAILS} emails...")
    emails = fetch_emails(service)

    print("[+] Loading transformer model...")
    tokenizer, model = load_model_and_tokenizer()

    print("[+] Classifying emails...\n")
    results = classify_emails(emails, tokenizer, model)

    for idx, (label, snippet) in enumerate(results, 1):
        print(f"--- Email {idx} ---")
        print(f"Prediction: {label}")
        print(f"Snippet   : {snippet.strip()}")
        print()