import json

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from data_preprocessing import preprocess_text, remove_hashtags_and_mentions


def tf_idf(file_name):
    # Load the json file
    with open(file_name, "r") as f:
        data = json.load(f)

    # Extract the 'desc' field
    descs = [entry["desc"] for entry in data]

    # Preprocess the text
    descs = [preprocess_text(desc) for desc in descs]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)

    # Fit and transform the data
    tfidf_matrix = vectorizer.fit_transform(descs)

    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    return tfidf_matrix


def bert(file_name):
    # Load the json file
    with open(file_name, "r") as f:
        data = json.load(f)

    # Extract the 'desc' field
    descs = [entry["desc"] for entry in data]

    # Extract the 'desc' field
    descs = [remove_hashtags_and_mentions(desc) for desc in descs]

    # Create a BERT model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode the text
    embeddings = model.encode(descs)

    print("BERT embeddings shape:", embeddings.shape)

    return embeddings


if __name__ == "__main__":
    # tf_idf("subset_10000.json")
    bert("subset_10000.json")
