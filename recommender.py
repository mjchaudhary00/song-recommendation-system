# recommender.py

import os
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "spotify_millsongdata.csv"
DATA_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
# NOTE: Replace with your dataset source if you have a direct CSV URL

def load_dataset():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "Dataset not found. Please download spotify_millsongdata.csv "
            "and place it in the project root directory."
        )
    return pd.read_csv(DATA_FILE)

df = load_dataset()

df = df[['song', 'artist', 'text']].dropna()
df['combined'] = df['song'] + " " + df['artist'] + " " + df['text']

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined'])

def recommend_songs(song_name, top_n=5):
    if song_name not in df['song'].values:
        return {"error": "Song not found in dataset"}

    idx = df[df['song'] == song_name].index[0]
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    recommendations = (
        pd.DataFrame({"song": df['song'], "score": scores})
        .sort_values("score", ascending=False)
        .drop_duplicates("song")
        .iloc[1: top_n + 1]["song"]
        .tolist()
    )

    return {
        "input_song": song_name,
        "recommended_songs": recommendations
    }
