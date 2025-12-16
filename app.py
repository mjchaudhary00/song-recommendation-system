from fastapi import FastAPI, Query
from recommender import recommend_songs

app = FastAPI(title="Song Recommendation System")

@app.get("/")
def home():
    return {"message": "Song Recommendation API is running"}

@app.get("/recommend")
def recommend(
    song_name: str = Query(...),
    top_n: int = Query(5, ge=1, le=10)
):
    return recommend_songs(song_name, top_n)
