import pandas as pd
import streamlit as st
import os
import pymongo
from surprise import Dataset, Reader, dump
from surprise.prediction_algorithms.matrix_factorization import SVD
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
headers = {
     "accept": "application/json",
     "Authorization": "Bearer " + st.secrets['tmdb_key']
}
pymongo_user = st.secrets['pymongo_user']


client = pymongo.MongoClient(pymongo_user)
db = client.TheMovieDatabase


# Function to get dataframe of DB from PyMongo Collection
def getPyMongoDB(requestedCollection):
    db_collection = db[requestedCollection]
    collection = db_collection.find()
    df_collection = pd.DataFrame(list(collection))
    return df_collection



#====================================== COLLABORATIVE FILTERING ======================================
# Function to get average rating of user
def getAverageRating(df_user):
    average_rating = df_user['rating'].mean()
    return average_rating

# Function to Return Dataframe used for SVD 
def getUserRatingSVDDF(df_user_initial, df_movies_db, username):
    # The function eliminates all ratings that are not found in the database
    df_user_initial["userId"] = username
    df_user_initial["movieId"] = ""
    for index, row in df_user_initial.iterrows():
        # Find the matching row in all_movies by imdb_id
        matching_row = df_movies_db[df_movies_db["imdb_id"] == row["IMDb_ID"]]
        if len(matching_row)> 0:
            df_user_initial.at[index, "movieId"] = matching_row["id"].values[0]

    df_user_SVD = df_user_initial[df_user_initial["movieId"] != ""]
    # Drop the specified columns
    columns_to_drop = ['letterboxd_id', 'liked', 'letterboxd_link', 'title','IMDb_ID']
    df_user_SVD = df_user_SVD.drop(columns=columns_to_drop) 
    return df_user_SVD

# Function to combine user rating and ratings dataset
def combineRatingsDF(df_ratings):
    # get columns userId, movieId, rating only for df_user_1_SVD
    df_ratings = df_ratings[['userId','movieId','rating']]
    df_ratings.reset_index(inplace=True, drop=True) # Reset Index
    # Get ratings dataset from PyMongo
    df_small_ratingsDB = getPyMongoDB("ratings_small")
    # drop _id and timestamp column of df_small_ratingsDB
    df_small_ratingsDB = df_small_ratingsDB.drop(columns=['_id', 'timestamp'])
    # Combine both user ratings and ratings dataset
    df_ratings_combined = pd.concat([df_ratings, df_small_ratingsDB], ignore_index=True)
    return df_ratings_combined

# Function to train SVD Model
def trainSVDModel(df_ratings):
    # Initialize a surprise reader object
    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0,5), skip_lines=1)
    # Load the data
    data = Dataset.load_from_df(df_ratings[['userId','movieId','rating']], reader=reader)
    trainset = data.build_full_trainset()

    # Initialize model
    svd = SVD()
    svd.fit(trainset)
    return svd

# Function to load SVD Model
def loadSVDModel(model_filename):
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    return loaded_model

# Function to Get SVD Recommendation
def getSVDRecommendations(data, df_movies, user_id, algo):
    # creating an empty list to store the recommended movie ids
    recommendations = []
    
    # creating an user item interactions matrix 
    user_movie_interactions_matrix = data.pivot(index='userId', columns='movieId', values='rating')
   
    # extracting those product ids which the user_id has not interacted yet
    non_interacted_movies = user_movie_interactions_matrix.loc[user_id][user_movie_interactions_matrix.loc[user_id].isnull()].index.tolist()
    
    # looping through each of the product ids which user_id has not interacted yet
    for item_id in non_interacted_movies:
        
        # predicting the ratings for those non interacted product ids by this user
        est = algo.predict(user_id, item_id).est
        
        # appending the predicted ratings
        recommendations.append((est, item_id))
        
    # sorting the predicted ratings in descending order
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations # returning sorted list



# Get SVD Results as Dataframe
def get_SVD_Dataframe(sorted_recommendations, df_movies, user_id, avgRating):
    df_with_SVD_ratings = df_movies.copy()
    # To store the predicted ratings for each movie
    predicted_ratings = {}
    
    for movie_info in sorted_recommendations:
        movie_id = movie_info[1]  # Extract movie id 
        rating = movie_info[0]  # Extract predicted SVD rating 
        
        # Add the predicted rating to the dictionary
        predicted_ratings[movie_id] = rating
    
    # Create a list to store the predicted ratings for all movies in the dataframe
    ratings = []
    
    # Iterate through all movies in the dataframe
    for index, row in df_with_SVD_ratings.iterrows():
        movie_id = row['id']
        
        # Check if the movie has a predicted rating in the dictionary
        if movie_id in predicted_ratings:
            # If there is a match, append the predicted rating
            ratings.append(predicted_ratings[movie_id])
        else:
            # If there is no match, append the average rating
            ratings.append(avgRating)
    
    # Add the SVD predicted ratings as a new column to the dataframe
    df_with_SVD_ratings['SVDRatings'] = ratings
    # Sort df_with_SVD_ratings according to SVDRatings
    df_with_SVD_ratings = df_with_SVD_ratings.sort_values(by=['SVDRatings'], ascending = False)
    return df_with_SVD_ratings

#====================================== CONTENT-BASED FILTERING ======================================
def getMovieKeyword(id):
    url = "https://api.themoviedb.org/3/movie/" + str(id) + "/keywords"
    response = requests.get(url, headers=headers)
    keywords_json = response.json()
    keyword_string = ""
    # check if keywords_json is not empty 
    if keywords_json['keywords']:
        # get keywords name from keywords_json
        for kw in keywords_json['keywords']:
            keyword_string += kw['name'] + " "    
    return keyword_string

def getTVKeyword(id):
    url = "https://api.themoviedb.org/3/tv/" + str(id) + "/keywords"
    response = requests.get(url, headers=headers)
    keywords_json = response.json()
    keyword_string = ""
    # check if keywords_json is empty
    if keywords_json['results']:
        # get keywords name from keywords_json
        for kw in keywords_json['results']:
            keyword_string += kw['name'] + " "
    return keyword_string

# function to get keywords using TMDB API for sorted df
def get_keywords(sorted_df):
    # for rows where media = 'movie', apply getMovieKeyword
    sorted_df['keywords'] = sorted_df.apply(lambda row: getMovieKeyword(row['tmdb_id']) if row['media'] == 'movie' else getTVKeyword(row['tmdb_id']), axis=1)
    return sorted_df

# Function to sort df according to ratings and liked
def sort_rated_df(df_user):
    # sort df according to ratings and liked and popularity
    sorted_df = df_user.sort_values(by=['rating', 'liked', 'popularity'], ascending =[False, False, False])
    # return top 10 movies in the sorted df
    sorted_df = sorted_df.head(10)
    sorted_df = get_keywords(sorted_df)
    return sorted_df


# Function to get overall description of movies in movie database
def get_CBF_description(df_movies):
    # Combine all relevant columns into one
    # fill NaN with empty string
    df_movies['overview'] = df_movies['overview'].fillna('') 
    df_movies['tagline'] = df_movies['tagline'].fillna('') 
    df_movies['genres'] = df_movies['genres'].fillna('') 
    df_movies['keywords'] = df_movies['keywords'].fillna('')

    df_movies['CBF_description'] = df_movies['overview'] + df_movies['tagline'] +df_movies['genres'] + df_movies['keywords']
    df_movies['CBF_description'] = df_movies['CBF_description'].fillna('')
    # remove words "id" and "name"
    df_movies['CBF_description'] = df_movies['CBF_description'].str.replace(r'id', '')
    df_movies['CBF_description'] = df_movies['CBF_description'].str.replace(r'name', '')
    
    # Clean the data
    stop_words = stopwords.words('english') # Remove stopwords
    df_movies['CBF_description'] = df_movies['CBF_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df_movies['CBF_description'] = df_movies['CBF_description'].str.replace(r'[^\w\s]', '') # remove punctuation
    df_movies['CBF_description'] = df_movies['CBF_description'].str.replace(r'\s+', ' ') # remove extra spaces
    df_movies['CBF_description'] = df_movies['CBF_description'].str.replace(r'\d+', '') # remove numbers
    df_movies['CBF_description'] = df_movies['CBF_description'].str.lower() # convert lowercase

    # Drop rows with CBF_description = ''
    df_movies = df_movies[df_movies['CBF_description'] != '']
    return df_movies

# Function to get overall description of movies in letterboxd df
def get_CBF_description_letterboxd(df_movies):
    # Combine all relevant columns into one
    df_movies['overview'] = df_movies['overview'].fillna('') 
    df_movies['genres'] = df_movies['genres'].fillna('') 
    # remove words "id" and "name" from genres
    df_movies['genres'] = df_movies['genres'].str.replace(r'id', '')
    df_movies['genres'] = df_movies['genres'].str.replace(r'name', '')

    df_movies['CBF_description'] = df_movies['overview'] + df_movies['genres'] + df_movies['keywords']
    df_movies['CBF_description'] = df_movies['CBF_description'].fillna('')
    
    # Clean the data
    stop_words = stopwords.words('english') # Remove stopwords
    df_movies['CBF_description'] = df_movies['CBF_description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    df_movies['CBF_description'] = df_movies['CBF_description'].str.replace(r'[^\w\s]', '') # remove punctuation
    df_movies['CBF_description'] = df_movies['CBF_description'].str.replace(r'\s+', ' ') # remove extra spaces
    df_movies['CBF_description'] = df_movies['CBF_description'].str.replace(r'\d+', '') # remove numbers
    df_movies['CBF_description'] = df_movies['CBF_description'].str.lower() # convert lowercase

    # Drop rows with CBF_description = ''
    df_movies = df_movies[df_movies['CBF_description'] != '']
    return df_movies

# Get top 5 genres of movies in df_user_letterboxd
def get_top_genres(df_user_letterboxd):
    genres = []
    for index, row in df_user_letterboxd.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres.append(genre)
    genres = ';'.join(genres) # convert list of lists to list of strings
    genres = genres.split(";")
    genres = nltk.FreqDist(genres)
    top_genres = genres.most_common(5)
    top_genres = [genre[0] for genre in top_genres]
    return top_genres

# Function to get CBF input
def get_filtered_CBF_input(df_svd_results, sorted_df, top_genres):
    # Sort df_svd_results according to SVDRatings
    df_svd_results = df_svd_results.sort_values(by=['SVDRatings','popularity','vote_average','vote_count'], ascending = [False, False, False, False])
    df_svd_results = df_svd_results[df_svd_results['genres'].str.contains('|'.join(top_genres))]
    # Drop rows of movies with vote_count < 50
    df_svd_results = df_svd_results[df_svd_results['vote_count'] > 50]
    # Drop rows of movies with same imdb_id values in df_user_letterboxd
    df_svd_results = df_svd_results[~df_svd_results['imdb_id'].isin(sorted_df['IMDb_ID'])]
    # Return top 1000 movies in df_svd_results
    df_svd_results = df_svd_results.head(200)
    return df_svd_results

# Function to get cosine similarity of movies in df_movies
def get_CBF_cosine_sim(df_movies, sorted_df):
    cosine_sim = []
    # loop through df_movies['CBF_description'] 
    # and get cosine similarity of each movie with all movies in sorted_df['CBF_description']
    for index_df_movies, row_df_movies in df_movies.iterrows():
        cosine_score = 0
        for index_sorted_df, row_sorted_df in sorted_df.iterrows():
            vectorizer = TfidfVectorizer(analyzer='word', min_df=0, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([row_df_movies['CBF_description'], row_sorted_df['CBF_description']])
            cosine_score += cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        cosine_sim.append(cosine_score)
    df_movies['cosine_sim'] = cosine_sim
    # Sort df_movies according to weightage of 20% SVDRatings and 80% cosine_sim
    df_movies['finalRatings'] = (df_movies['cosine_sim'] * 30) + df_movies['SVDRatings']
    # print max and min cosine_sim
    print("Max cosine_sim: ", df_movies['cosine_sim'].max())
    print("Min cosine_sim: ", df_movies['cosine_sim'].min())
    #df_movies['SVDRatings'] = df_movies['SVDRatings'] * 0.2
    #df_movies['cosine_sim'] = df_movies['cosine_sim'] + df_movies['SVDRatings']
    df_cbf_results = df_movies.sort_values(by='finalRatings', ascending = False)
    return df_cbf_results
