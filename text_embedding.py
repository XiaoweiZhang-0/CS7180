import json

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from data_preprocessing import preprocess_text, remove_hashtags_and_mentions


def tf_idf(file_name, column_name):
    # Load the json file
    with open(file_name, "r") as f:
        data = json.load(f)
    if column_name == "challenges":
        # print all the keys in data
        # for entry in data:
        #     print(entry['challenges'])
        # get the chanllenge title from the challenges list
        column_data = []
        for entry in data:
            if "challenges" not in entry:
                column_data.append("")
            else:
                challenges = entry["challenges"]
                if challenges:
                    column_data.append(
                        " ".join([challenge["title"] for challenge in challenges])
                    )
        if len(column_data) != len(data):
            print("Column data size does not match data size")
            return
    # confirm the column_data is of size len(data)
    else:
        # if column_name is a nested field, extract it
        if "." in column_name:
            column_name_list = column_name.split(".")
        # Extract the nested field if any
            column_data = [entry[column_name_list[0]][column_name_list[1]] for entry in data]
        else:
            # Extract the field
            column_data = [entry[column_name] for entry in data]          

    # Preprocess the text
    column_data = [preprocess_text(data) for data in column_data]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)

    # Fit and transform the data
    tfidf_matrix = vectorizer.fit_transform(column_data)

    print("TF-IDF matrix shape:", tfidf_matrix.shape)

    return tfidf_matrix


def bert(file_name, column_name):
    # Load the json file
    with open(file_name, "r") as f:
        data = json.load(f)

    if column_name == "challenges":
        # print all the keys in data
        # for entry in data:
        #     print(entry['challenges'])
        # get the chanllenge title from the challenges list
        column_data = []
        for entry in data:
            if "challenges" not in entry:
                column_data.append("")
            else:
                challenges = entry["challenges"]
                if challenges:
                    column_data.append(
                        " ".join([challenge["title"] for challenge in challenges])
                    )
        if len(column_data) != len(data):
            print("Column data size does not match data size")
            return
    # confirm the column_data is of size len(data)
    else:
        # if column_name is a nested field, extract it
        if "." in column_name:
            column_name_list = column_name.split(".")
        # Extract the nested field if any
            column_data = [entry[column_name_list[0]][column_name_list[1]] for entry in data]
        else:
            # Extract the 'desc' field
            column_data = [entry[column_name] for entry in data]  

    # Extract the 'desc' field
    column_data = [remove_hashtags_and_mentions(data) for data in column_data]

    # Create a BERT model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode the text
    embeddings = model.encode(column_data)

    print("BERT embeddings shape:", embeddings.shape)

    return embeddings

# Input: the text in a field (e.g., 'desc', 'author.signature', etc.)
# Note: format of challenges should be space separated string of challenge titles
# Output: BERT embeddings
def bert_for_prediction(text):
    # Create a BERT model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Preprocess the text
    data = remove_hashtags_and_mentions(text)

    # Encode the text
    embeddings = model.encode(data)

    print("BERT embeddings shape:", embeddings.shape)

    return embeddings

if __name__ == "__main__":
    tf_idf("subset_10000.json")
    # bert("subset_10000.json")
