import streamlit as st
import pandas as pd
from recommender import (
    get_high_rated,
    get_most_rated,
    similar_movies,
    movies
)

@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()


st.title("🎬 Movie Recommendation System")

option = st.selectbox(
    "Choose Recommendation Type",
    ["High Rated Movies",
     "Most Rated Movies",
     "Similar Movies (Collaborative Filtering)"]
)

n = st.slider("Number of Movies", 1, 20, 5)

if option == "High Rated Movies":
    st.write(get_high_rated(n)[['title','genres','avg_rating']])

elif option == "Most Rated Movies":
    st.write(get_most_rated(n)[['title','genres','total_ratings']])

elif option == "Similar Movies (Collaborative Filtering)":
    movie_name = st.selectbox("Select Movie", movies['title'].values)
    
    movie_id = movies[movies['title'] == movie_name]['movieId'].values[0]
    
    if st.button("Recommend"):
        st.write(similar_movies(movie_id, n))
