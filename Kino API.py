import re
import string

import fastapi
import joblib
import nltk
import pymorphy3

app = fastapi.FastAPI()

def preprocess_text(text: str) -> str:
    punctuation = set(string.punctuation) - { "(", ")" }
    stopwords = nltk.corpus.stopwords.words("russian") + [
        "который", "весь", "всё", "это", "свой", "мочь", "история",
        "год", "человек", "самый", "день", "молодой", "хороший"
    ]

    morph = pymorphy3.MorphAnalyzer()

    text =  "".join([char for char in text if char not in punctuation])
    text = "".join([char for char in text if not char.isdigit()])
    text = re.sub(r"[a-z]", "", text)
    text = re.sub(r"\s+", " ", text.strip())
    tokenized_text = re.split(r"\W+", text)
    tokenized_text = [morph.parse(word)[0].normal_form for word in tokenized_text]
    tokenized_text = [word for word in tokenized_text if word not in stopwords]
    return " ".join(tokenized_text)

def predict_cluster(text):
    with open('KNeighborsClassifier.joblib', 'rb') as file:
        model = joblib.load(file)

    with open('tfidf_vectorizer.joblib', 'rb') as file:
        vectorizer = joblib.load(file)

    with open('nmf_model.joblib', 'rb') as file:
        nmf_model = joblib.load(file)

    tfidf_matrix = vectorizer.transform([preprocess_text(text)])

    W = nmf_model.transform(tfidf_matrix)

    prediction = model.predict(W)
    probabilities = model.predict_proba(W)[0]

    mapping = {
        0: "Городской криминал / Драма",
        1: "Дружба / Любовь / Испытания",
        2: "Космические путешествия / Приключения / Судьба",
        3: "Магия / Волшебство / Эпическая борьба",
        4: "Война / Мужская доблесть",
        5: "Любовь / Семейные отношения",
        6: "Современная жизнь / Преступность / Социальные драмы",
    }

    probabilities_dict = { mapping[i]: float(probabilities[i]) for i in range(len(probabilities)) }

    return {
        #"predict" : mapping[prediction[0]],
        "probabilities" : probabilities_dict
    }

@app.post("/predict")
def predict_class(text: str):
    return predict_cluster(text)
