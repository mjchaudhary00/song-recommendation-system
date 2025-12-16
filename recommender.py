# recommender.py
# Content-based Song Recommendation System (Memory-safe)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset (CSV must be in same folder)
df = pd.read_csv("spotify_millsongdata.csv")

# Keep only required columns
df = df[['song', 'artist', 'text']].dropna()

# Combine song name, artist, and lyrics into one text feature
df['combined_text'] = (
    df['song'].astype(str) + " " +
    df['artist'].astype(str) + " " +
    df['text'].astype(str)
)

# Reset index for safe alignment
df.reset_index(drop=True, inplace=True)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_text'])

def recommend_songs(song_name: str, top_n: int = 5):
    """
    Returns top_n unique recommended songs for a given song_name
    """

    # Check if song exists
    if song_name not in df['song'].values:
        return {"error": "Song not found in database"}

    # Get index of selected song
    song_index = df[df['song'] == song_name].index[0]

    # Get TF-IDF vector of selected song
    song_vector = tfidf_matrix[song_index]

    # Compute cosine similarity (one song vs all)
    similarity_scores = cosine_similarity(song_vector, tfidf_matrix).flatten()

    # Create similarity DataFrame
    similarity_df = pd.DataFrame({
        "song": df['song'],
        "score": similarity_scores
    })

    # Remove the input song itself
    similarity_df = similarity_df[similarity_df['song'] != song_name]

    # Sort by similarity, remove duplicate titles, and select top_n
    recommendations = (
        similarity_df
        .sort_values(by="score", ascending=False)
        .drop_duplicates(subset="song")
        .head(top_n)["song"]
        .tolist()
    )

    return {
        "input_song": song_name,
        "recommended_songs": recommendations
    }
