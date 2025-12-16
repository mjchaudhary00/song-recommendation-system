# recommender.py

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "spotify_millsongdata.csv"

df = None
tfidf = None
tfidf_matrix = None


def load_dataset_once():
    global df, tfidf, tfidf_matrix

    if df is not None:
        return

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "Dataset not found. Please download spotify_millsongdata.csv "
            "and place it in the project root directory."
        )

    df = pd.read_csv(DATA_FILE)
    df = df[['song', 'artist', 'text']].dropna()
    df['combined'] = df['song'] + " " + df['artist'] + " " + df['text']

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['combined'])


def recommend_songs(song_name, top_n=5):
    try:
        load_dataset_once()
    except FileNotFoundError as e:
        return {"error": str(e)}

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
