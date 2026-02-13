import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

merged = pd.merge(ratings, movies, on="movieId")

# High Rated Movies
rate = (
    merged.groupby(['movieId','title','genres'])
    .agg(total_ratings=('rating','count'),
         avg_rating=('rating','mean'))
    .reset_index()
)

rate = rate.sort_values(by='avg_rating', ascending=False)

def get_high_rated(n):
    return rate.head(n)


# Most Rated Movies
most = (
    merged.groupby(['movieId','title','genres'])
    .agg(total_ratings=('rating','count'),
         avg_rating=('rating','mean'))
    .reset_index()
)

most = most.sort_values(by='total_ratings', ascending=False)

def get_most_rated(n):
    return most.head(n)


# Collaborative Filtering
user_movie_matrix = merged.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

movie_similarity = cosine_similarity(user_movie_matrix.T)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)

def similar_movies(movie_id, n):
    similar = movie_similarity_df[movie_id].sort_values(ascending=False)
    similar = similar.iloc[1:n+1]
    return movies[movies['movieId'].isin(similar.index)][['title','genres']]
