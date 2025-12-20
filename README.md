# 🎬 Movie Recommendation System (Content-Based)

A simple **content-based movie recommender** built with **Python + scikit-learn** and deployed as a **Streamlit web app**.  
It recommends movies based on textual similarity using **TF-IDF** and **cosine similarity** (TMDB 5000 dataset).

---

## ✅ Features
- Content-based recommendations (no user ratings needed)
- Uses **overview + genres + keywords + top cast + director**
- Interactive UI with **Streamlit**
- Adjustable number of recommendations

---

## 🧠 How it works
1. Load TMDB movies + credits datasets
2. Extract and clean:
   - Genres, Keywords
   - Top 3 cast members
   - Director from crew
3. Build a `tags` text field per movie
4. Vectorize text using **TF-IDF**
5. Compute similarity using **cosine similarity**
6. Return the top-N most similar movies

---

## 🧰 Tech Stack
- Python
- pandas
- scikit-learn
- Streamlit

---

## 📂 Project Structure
```text
ai-movie-recommender/
├── app/
│   └── app.py
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── notebooks/
│   └── 01_exploration.ipynb
├── requirements.txt
└── README.md
