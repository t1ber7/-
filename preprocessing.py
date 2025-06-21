import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

stemmer = SnowballStemmer("russian")
stop_words = set(stopwords.words("russian"))


def preprocess_text_news(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  # удаление HTML-тегов
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    tokens = word_tokenize(text, language="russian")

    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]

    return " ".join(processed_tokens)