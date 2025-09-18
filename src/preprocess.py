import re

def clean_email(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)               # strip HTML tags
    text = re.sub(r"https?://\\S+", " <URL> ", text)   # mask links
    text = re.sub(r"\\b[\\w\\.-]+@[\\w\\.-]+\\.\\w+\\b", " <EMAIL> ", text)
    text = re.sub(r"\\+?\\d[\\d\\-\\s]{7,}\\d", " <PHONE> ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text