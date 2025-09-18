import os
import json
from flask import Flask, redirect, request, session, url_for, render_template
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from base64 import urlsafe_b64decode
from email import message_from_bytes

from src.classify_baseline import classify_emails as baseline_classifier
from src.classify_transformer import classify_emails as transformer_classifier
from src.config import MODEL_TYPE, MAX_EMAILS

# === App Config ===
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_key")  # Use env for production

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# === Auth Flow ===
def get_flow():
    client_config = json.loads(os.environ["CLIENT_SECRET"])
    return Flow.from_client_config(
        client_config=client_config,
        scopes=SCOPES,
        redirect_uri=url_for("oauth2callback", _external=True)
    )

def get_gmail_service():
    if "credentials" not in session:
        return None

    creds = Credentials.from_authorized_user_info(session["credentials"], SCOPES)
    return build("gmail", "v1", credentials=creds)


# === Routes ===
@app.route("/")
def index():
    return redirect("/login")


@app.route("/login")
def login():
    flow = get_flow()
    auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true")
    session["state"] = state
    return redirect(auth_url)


@app.route("/oauth2callback")
def oauth2callback():
    flow = get_flow()
    flow.fetch_token(authorization_response=request.url)

    credentials = flow.credentials
    session["credentials"] = {
        "token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_uri": credentials.token_uri,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "scopes": credentials.scopes,
    }

    return redirect(url_for("inbox"))


@app.route("/inbox")
def inbox():
    service = get_gmail_service()
    if not service:
        return redirect("/login")

    messages = service.users().messages().list(userId="me", maxResults=MAX_EMAILS).execute().get("messages", [])
    email_bodies = []

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        payload = msg_data.get("payload", {})
        parts = payload.get("parts", [])
        body = ""

        for part in parts:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data")
                if data:
                    try:
                        decoded = urlsafe_b64decode(data.encode("UTF-8"))
                        body = message_from_bytes(decoded).get_payload()
                        break
                    except Exception:
                        continue

        if not body:
            body = msg_data.get("snippet", "")

        email_bodies.append(body)

    # Use either baseline or transformer
    if MODEL_TYPE == "transformer":
        results = transformer_classifier(email_bodies)
    else:
        results = baseline_classifier(email_bodies)

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)