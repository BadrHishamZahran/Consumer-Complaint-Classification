import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)

def clean_text(text: str) -> str:
    # Exact preprocessing from the training notebook.
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [word for word in text.split() if word not in STOP_WORDS]
    return " ".join(tokens)
