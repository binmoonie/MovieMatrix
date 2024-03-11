import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
movies_df = pd.read_csv('ml-latest-small/movies.csv')
rating_df = pd.read_csv('ml-latest-small/ratings.csv')

data = pd.merge(movies_df, rating_df, on='movieId')

# Calculate average rating and number of ratings per movie
ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
ratings['number_of_ratings'] = data.groupby('title')['rating'].count()

# Create a movie-genre matrix
genre_matrix = pd.get_dummies(data['genres'].str.split('|', expand=True), prefix='', prefix_sep='')

# Combine genres for each movie
genre_matrix['combined_genres'] = genre_matrix.apply(lambda row: '|'.join(row.index[row == 1]), axis=1)

# Merge genre information back into the main data
data = pd.concat([data, genre_matrix['combined_genres']], axis=1)

# Create Pivot Table
movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')


def recommend_movies(title, movie_matrix=movie_matrix, genre_matrix=genre_matrix):
    try:
        # get the movieId for the input movie title
        movie_id = movies_df[movies_df['title'] == title]['movieId'].values[0]

        # get the genres of the input movie
        input_genres = genre_matrix.loc[movie_id, 'combined_genres']

        # find movies with similar genres
        similar_movies = genre_matrix[genre_matrix['combined_genres'].str.contains(input_genres)]

        # Calculate the correlation between input movie and similar movies
        movie_ratings = movie_matrix[movie_id]
        similar_ratings = movie_matrix[similar_movies.index]
        similar_corr = similar_ratings.corrwith(movie_ratings)

        # Drop NaN values and sort by correlation
        corr_movies = pd.DataFrame(similar_corr, columns=['correlation'])
        corr_movies.dropna(inplace=True)

        return corr_movies.sort_values('correlation', ascending=False).head(20)

    except KeyError:
        return f"Movie '{title}' not found."

print(recommend_movies('Toy Story (1995)'))
