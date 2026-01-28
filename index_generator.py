# index_generator.py

import json
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_index_mapping(documents, max_features=10000, output_path="generated_index_to_word.json"):
    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(documents)

    # Get word to index mapping
    word_to_index = vectorizer.vocabulary_

    # Invert to get index to word mapping
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    # Save to file
    with open(output_path, "w") as f:
        json.dump(index_to_word, f)

    print(f"Index-to-word mapping saved to: {output_path}")

# Example usage:
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("people_wiki.csv")
    documents = df["text"].tolist()
    generate_index_mapping(documents)
