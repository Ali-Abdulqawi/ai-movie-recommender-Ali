import ast
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

@st.cache_data
def load_data():
    base = Path(__file__).resolve().parent.parent  # project root
    data_dir = base / "data"

    movies = pd.read_csv(data_dir / "tmdb_5000_movies.csv")
    credits = pd.read_csv(data_dir / "tmdb_5000_credits.csv")

    df = movies.merge(credits, on="title")
    df = df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

    def parse_names(text):
        try:
            items = ast.literal_eval(text)
            return [i["name"] for i in items]
        except:
            return []

    def get_director(text):
        try:
            crew_list = ast.literal_eval(text)
            for person in crew_list:
                if person.get("job") == "Director":
                    return [person.get("name")]
            return []
        except:
            return []

    def clean_list(lst):
        return [s.replace(" ", "").lower() for s in lst]

    df["genres"] = df["genres"].apply(parse_names).apply(clean_list)
    df["keywords"] = df["keywords"].apply(parse_names).apply(clean_list)
    df["cast"] = df["cast"].apply(parse_names).apply(lambda x: x[:3]).apply(clean_list)
    df["crew"] = df["crew"].apply(get_director).apply(clean_list)

    df["overview"] = df["overview"].fillna("").astype(str).str.lower()

    df["tags"] = (
        df["overview"] + " "
        + df["genres"].apply(lambda x: " ".join(x)) + " "
        + df["keywords"].apply(lambda x: " ".join(x)) + " "
        + df["cast"].apply(lambda x: " ".join(x)) + " "
        + df["crew"].apply(lambda x: " ".join(x))
    )

    final_df = df[["movie_id", "title", "tags"]].copy()
    return final_df

@st.cache_resource
def build_model(final_df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    X = tfidf.fit_transform(final_df["tags"])
    sim = cosine_similarity(X)
    return sim

def recommend(final_df: pd.DataFrame, sim, title: str, top_n: int = 10):
    title_clean = title.lower().strip()
    titles_lower = final_df["title"].str.lower()

    if title_clean not in set(titles_lower):
        suggestions = final_df[titles_lower.str.contains(title_clean, na=False)]["title"].head(10).tolist()
        return None, suggestions

    idx = final_df[titles_lower == title_clean].index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1: top_n + 1]
    recs = final_df.iloc[[i for i, _ in scores]][["title"]]
    return recs, None


st.title("🎬 Movie Recommendation System")
st.caption("Content-based recommender using TF-IDF + cosine similarity (TMDB 5000 dataset).")

final_df = load_data()
sim = build_model(final_df)

movie_list = sorted(final_df["title"].unique().tolist())
selected = st.selectbox("Choose a movie", movie_list, index=movie_list.index("Avatar") if "Avatar" in movie_list else 0)

top_n = st.slider("How many recommendations?", min_value=5, max_value=20, value=10, step=1)

if st.button("Recommend"):
    recs, suggestions = recommend(final_df, sim, selected, top_n=top_n)

    if recs is not None:
        st.subheader("Recommended movies")
        for i, t in enumerate(recs["title"].tolist(), start=1):
            st.write(f"{i}. {t}")
    else:
        st.error("Movie title not found.")
        if suggestions:
            st.write("Did you mean:")
            st.write(suggestions)

