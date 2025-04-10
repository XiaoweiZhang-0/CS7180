import json

from sklearn.feature_extraction.text import TfidfVectorizer

from data_preprocessing import preprocess_text


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


if __name__ == "__main__":
    tf_idf("subset_10000.json")
