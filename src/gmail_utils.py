from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def get_gmail_service(credentials_dict):
    creds = Credentials.from_authorized_user_info(credentials_dict)
    return build('gmail', 'v1', credentials=creds)

def fetch_emails(service, max_results=10):
    messages = []
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    for msg in results.get('messages', []):
        data = service.users().messages().get(userId='me', id=msg['id']).execute()
        snippet = data.get('snippet', '')
        messages.append(snippet)
    return messages