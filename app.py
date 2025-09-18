import os
from flask import Flask, redirect, url_for, session, render_template, request
from flask_session import Session
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import base64
from email import message_from_bytes
# Import your models
from src.classify_baseline import classify_emails as classify_baseline
from src.classify_transformer import load_model_and_tokenizer, classify_emails as classify_transformer

app = Flask(__name__)
app.secret_key = "REPLACE_WITH_SOMETHING_SECRET"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# === Constants ===
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
MODEL_TYPE = "transformer"  # or "baseline"
NUM_EMAILS = 10


# === OAuth2 Flow ===
def get_flow():
    return Flow.from_client_secrets_file(
        "client_secret.json",
        scopes=SCOPES,
        redirect_uri=url_for("oauth2callback", _external=True)
    )


@app.route("/")
def index():
    if "credentials" not in session:
        return redirect(url_for("login"))
    return redirect(url_for("inbox"))


@app.route("/login")
def login():
    flow = get_flow()
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline", include_granted_scopes="true")
    return redirect(auth_url)


@app.route("/oauth2callback")
def oauth2callback():
    flow = get_flow()
    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials
    session["credentials"] = creds_to_dict(creds)
    return redirect(url_for("inbox"))


@app.route("/inbox")
def inbox():
    if "credentials" not in session:
        return redirect(url_for("login"))

    creds = Credentials.from_authorized_user_info(session["credentials"])
    service = build("gmail", "v1", credentials=creds)

    # Fetch last N emails
    messages = service.users().messages().list(userId="me", maxResults=NUM_EMAILS).execute().get("messages", [])
    email_bodies = []

    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        payload = msg_data.get("payload", {})
        parts = payload.get("parts", [])
        body = ""

        for part in parts:
            if part.get("mimeType") == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    decoded_bytes = base64.urlsafe_b64decode(data.encode("UTF-8"))
                    body = message_from_bytes(decoded_bytes).get_payload()
                    break

        if not body:
            body = msg_data.get("snippet", "")

        email_bodies.append(body)

    # === Choose classification method ===
    if MODEL_TYPE == "baseline":
        results = classify_baseline(email_bodies)
    else:
        tokenizer, model = load_model_and_tokenizer()
        results = classify_transformer(email_bodies, tokenizer, model)

    return render_template("index.html", results=results)


def creds_to_dict(creds):
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": creds.scopes,
    }


if __name__ == "__main__":
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"  # Only for local testing (use HTTPS in production!)
    app.run(debug=True)
