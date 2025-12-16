# 🎵 Song Recommendation System

**FastAPI | Machine Learning | NLP | Cloud Deployment**

A production-ready **content-based song recommendation system** that uses **Natural Language Processing (NLP)** and **Machine Learning** to recommend songs based on lyrical and metadata similarity. The system exposes recommendations via a **FastAPI REST API** and is deployed on **Render**.

---

## 🚀 Live Demo

* **Live API:**
  [https://song-recommendation-system-1-h0d6.onrender.com/](https://song-recommendation-system-1-h0d6.onrender.com/)

* **Health Check:**

  ```
  GET /
  ```

Response:

```json
{
  "message": "Song Recommendation API is running"
}
```

---

## 📌 Project Overview

This project demonstrates an **end-to-end ML system** including:

* Data preprocessing and feature engineering
* TF-IDF based text vectorization
* Cosine similarity for recommendation
* REST API development using FastAPI
* Cloud deployment with proper production safeguards

The system is designed to **start safely even when large datasets are excluded**, reflecting real-world deployment constraints.

---

## 🎯 Key Features

* Content-based song recommendation engine
* NLP-driven similarity modeling
* RESTful API for real-time access
* Lazy dataset loading to avoid deployment crashes
* Cloud-hosted and publicly accessible
* Clean GitHub repository with size-safe data handling

---

## 🧠 Recommendation Approach

* **Model Type:** Content-Based Filtering
* **Text Features:** Song name + artist + lyrics
* **Vectorization:** TF-IDF (max 5000 features, English stopwords)
* **Similarity Metric:** Cosine Similarity

This approach does **not** require user interaction history and avoids cold-start problems.

---

## 🛠️ Tech Stack

| Category         | Tools                |
| ---------------- | -------------------- |
| Language         | Python 3.13          |
| Data Handling    | Pandas, NumPy        |
| Machine Learning | Scikit-learn         |
| NLP              | TF-IDF Vectorization |
| API Framework    | FastAPI              |
| ASGI Server      | Uvicorn              |
| Version Control  | Git, GitHub          |
| Deployment       | Render               |

---

## 🌐 API Endpoints

### 🔹 Health Check

```
GET /
```

### 🔹 Song Recommendation

```
GET /recommend
```

**Query Parameters:**

* `song_name` (string, required)
* `top_n` (integer, optional, default = 5)

**Example:**

```
/recommend?song_name=Andante&top_n=5
```

**Response (if dataset not present):**

```json
{
  "error": "Dataset not found. Please download spotify_millsongdata.csv and place it in the project root directory."
}
```

---

## 📂 Dataset Information

* **Dataset:** Spotify Million Song Dataset (lyrics-based)
* **File:** `spotify_millsongdata.csv`
* **Size:** ~72 MB

### ⚠️ Important Note

The dataset is **not included** in this repository due to GitHub file size limits.

This is intentional and handled gracefully in the code.

---

## ▶️ Running Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/mjchaudhary00/song-recommendation-system.git
cd song-recommendation-system
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add dataset

Download `spotify_millsongdata.csv` from Kaggle and place it in the project root directory.

### 4️⃣ Run the API

```bash
python -m uvicorn app:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

---

## ☁️ Deployment

* **Platform:** Render (Web Service)
* **Start Command:**

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 10000
```

* Dataset loading is **lazy**, ensuring the service stays live even if the dataset is missing.

---

## 🧪 Testing

* Health endpoint verified
* API starts without crashing
* Graceful error handling for missing dataset
* Successful cloud deployment with Uvicorn

---

## ⚠️ Limitations

* Requires local dataset for full recommendations
* Content-based filtering only
* No user personalization or feedback loop

---

## 🔮 Future Improvements

* Cloud storage for dataset (AWS S3 / GCP)
* Collaborative filtering
* Caching TF-IDF vectors
* Frontend UI (React / Streamlit)
* Authentication & rate limiting

---

## 👤 Author

**Mj Chaudhary**

* GitHub: [https://github.com/mjchaudhary00](https://github.com/mjchaudhary00)

---

## 📄 License

This project is intended for **educational and demonstration purposes**.



Say the word.
