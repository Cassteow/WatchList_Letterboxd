import pandas as pd
import nltk
import requests
from letterboxd_processing import getGenresFromID, getTVGenreList, getMovieGenreList
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

headers = {
    "accept": "application/json",
    "Authorization": "Bearer "+ st.secrets['tmdb_key']
}

# Function to check validity of netflixhistory df
def check_netflixhistory(netflixhistory):
    # check if there are Title and Date columns only
    if len(netflixhistory.columns) != 2:
        return "invalid"
    # check if column name is Title and Date
    elif netflixhistory.columns[0] != 'Title' or netflixhistory.columns[1] != 'Date':
        return "invalid"
    # check if rows are more than 5
    elif len(netflixhistory) < 5:
        return "insufficient"
    else:
        return "valid"

# Function to process netflixhistory df
def process_netflixhistory(netflixhistory):
    # Remove values after : in Title column
    netflixhistory['Title'] = netflixhistory['Title'].str.split(':', expand=True)[0]

    # Remove duplicate rows
    netflixhistory = netflixhistory.drop_duplicates()

    # create new title_api column where white space is replaced with %20
    netflixhistory['title_api'] = netflixhistory['Title'].str.replace(' ', '%20')

    # replace â€™s with '
    netflixhistory['title_api'] = netflixhistory['title_api'].str.replace('â€™s', "'s")

    # add column in calculating frequency of movie watched
    netflixhistory['frequency'] = netflixhistory.groupby('Title')['Title'].transform('count')

    # remove duplication in title column
    netflixhistory = netflixhistory.drop_duplicates(subset=['Title'])

    # convert date column to datetime
    netflixhistory['Date'] = pd.to_datetime(netflixhistory['Date'])

    # rearrange netflixhistory columns in order of latest date and frequency
    netflixhistory = netflixhistory.sort_values(by=['frequency', 'Date'], ascending=[False, False])
    return netflixhistory

def getTVDetailsNetflix(searchString):
    # Create URL
    tv_url = "https://api.themoviedb.org/3/search/tv?query=" + searchString + "&include_adult=false&page=1"
    # Get Response
    tv_response = requests.get(tv_url, headers=headers)
    # Convert Response to JSON
    tv_response_json = tv_response.json()

    # check if response_json is not empty
    if len(tv_response_json['results']) != 0:
        # Get First Result
        first_result = tv_response_json['results'][0]
        return first_result
    else:
        return "Not TV"
    
def getFilmDetailsNetflix(searchString):
    # Create URL
    movie_url = "https://api.themoviedb.org/3/search/movie?query=" + searchString + "&include_adult=false&page=1"
    # Get Response
    movie_response = requests.get(movie_url, headers=headers)
    # Convert Response to JSON
    movie_response_json = movie_response.json()

    # check if response_json is not empty
    if len(movie_response_json['results']) != 0:
        first_result = movie_response_json['results'][0]
        return first_result
    else:
        return "Not Movie"
    
# Function to get IMDb ID from TMDB API
def getTVIMDbID(id):
    url = "https://api.themoviedb.org/3/tv/"+id+"/external_ids"
    response = requests.get(url, headers=headers)
    response_json = response.json()
    # check if response_json imdb_id is not empty
    if response_json['imdb_id'] != "":
        return response_json['imdb_id']
    else:
        return "None"
    
def getFilmIMDbID(id):
    url = "https://api.themoviedb.org/3/movie/"+id+"/external_ids"
    response = requests.get(url, headers=headers)
    response_json = response.json()
    # check if response_json imdb_id is not empty
    if response_json['imdb_id'] != "":
        return response_json['imdb_id']
    else:
        return "None"

# Function to apply rating rules for movies
def applyMovieRating(df_movie, top3_genre, least_genre):
    # Frequency > 5, vote_average > 7, rating 5
    df_movie.loc[(df_movie['frequency'] > 5) & (df_movie['vote_average'] >= 7), 'rating'] = 5
    # Frequency >2 in top 3 genre, vote_average > 6.5, rating 4.5
    df_movie.loc[(df_movie['frequency'] > 2) & (df_movie['genres'].str.contains('|'.join(top3_genre))) & (df_movie['vote_average'] >= 6.5), 'rating'] = 4.5
    # In top 3 genre, vote_average > 7, rating 4
    df_movie.loc[(df_movie['genres'].str.contains('|'.join(top3_genre))) & (df_movie['vote_average'] >= 7), 'rating'] = 4
    # Frequency >3, rating 4
    df_movie.loc[(df_movie['frequency'] >3), 'rating'] = 4
    # Frequency >1 in top 3 genre, vote_average > 6.5, rating 3.5
    df_movie.loc[(df_movie['frequency'] > 1) & (df_movie['genres'].str.contains('|'.join(top3_genre))) & (df_movie['vote_average'] >= 6.5), 'rating'] = 3.5
    # Vote_average > 8, rating 3.5
    df_movie.loc[(df_movie['vote_average'] >= 8), 'rating'] = 3.5
    # Frequency 1, in least fav genre, rating 2
    df_movie.loc[(df_movie['frequency'] == 1) & (df_movie['genres'].str.contains(least_genre)) & (df_movie['vote_average'] <= 5), 'rating'] = 2
    # Others rating 3
    df_movie.loc[(df_movie['rating'].isnull()), 'rating'] = 3
    return df_movie

# Function to apply rating rules for tv
def applyTVRating(df_tv, top3_genre, least_genre):
    # Frequency > 40, rating 5
    df_tv.loc[(df_tv['frequency'] > 40), 'rating'] = 5
    # Frequency > 14 in top 3 genre, rating 5
    df_tv.loc[(df_tv['frequency'] > 14) & (df_tv['genres'].str.contains('|'.join(top3_genre))) & (df_tv['vote_average'] >= 7.5), 'rating'] = 5
    # In top 3 genre, vote_average>7, rating 4
    df_tv.loc[(df_tv['genres'].str.contains('|'.join(top3_genre))) & (df_tv['vote_average'] >= 7), 'rating'] = 4
    # Frequency > 20, rating 4
    df_tv.loc[(df_tv['frequency'] > 20), 'rating'] = 4
    # Frequency > 10 in top 3 genre, rating 3.5
    df_tv.loc[(df_tv['frequency'] > 10) & (df_tv['genres'].str.contains('|'.join(top3_genre))), 'rating'] = 3.5
    # Frequency <5, vote_average < 5, rating 2.5
    df_tv.loc[(df_tv['frequency'] < 5) & (df_tv['vote_average'] <= 5), 'rating'] = 2.5
    # Frequency 1, in least fav genre, rating 2
    df_tv.loc[(df_tv['frequency'] == 1) & (df_tv['genres'].str.contains(least_genre)) & (df_tv['vote_average'] <= 5), 'rating'] = 2
    # Others rating 3
    df_tv.loc[(df_tv['rating'].isnull()), 'rating'] = 3
    return df_tv

# Get top and least fav genres of movies in df
def get_genres(df):
    genres = []
    for index, row in df.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres.append(genre)
    genres = ';'.join(genres) # convert list of lists to list of strings
    genres = genres.split(";")
    genres = nltk.FreqDist(genres)
    top_genres = genres.most_common(3)
    top_genres = [genre[0] for genre in top_genres]
    least_genre = genres.most_common()[-1][0]
    return top_genres, least_genre

# Function to get top 70 netflix movies
def getNetflixMetadata(df_netflixhistory):
    # get top 70
    df_netflixhistory_top = df_netflixhistory.head(70)
    df_netflixhistory_top = df_netflixhistory_top.reset_index(drop=True)

    with st.spinner('Getting Netflix TV Data'):
        # add column for tv details in netflixhistory_analysis
        df_netflixhistory_top['tv_details'] = df_netflixhistory_top['title_api'].apply(getTVDetailsNetflix)

        # get rows with tv_details = Not TV as a new dataframe netflixhistory_analysis_movie
        netflixhistory_analysis_movie = df_netflixhistory_top[df_netflixhistory_top['tv_details'] == 'Not TV']

        # drop rows with tv_details = Not TV from netflixhistory_analysis
        netflixhistory_analysis_tv = df_netflixhistory_top[df_netflixhistory_top['tv_details'] != 'Not TV']

        # for TV 
        tVGenreList = getTVGenreList()
        # add relevant column in netflixhistory_analysis_tv extracted from tv_details dict value column
        netflixhistory_analysis_tv['id'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['id']) 
        netflixhistory_analysis_tv['IMDb_ID'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: getTVIMDbID(str(x['id'])))
        netflixhistory_analysis_tv['adult'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['adult']) 
        netflixhistory_analysis_tv['title'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['name'])
        netflixhistory_analysis_tv['original_language'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['original_language'])
        netflixhistory_analysis_tv['overview'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['overview'].replace('\n', ' '))
        netflixhistory_analysis_tv['vote_average'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['vote_average'])
        netflixhistory_analysis_tv['vote_count'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['vote_count'])
        netflixhistory_analysis_tv['popularity'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['popularity'])
        netflixhistory_analysis_tv['release_date'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['first_air_date'])
        netflixhistory_analysis_tv['genres'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: getGenresFromID(tVGenreList, x['genre_ids']))
        netflixhistory_analysis_tv['poster_path'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['poster_path'])
        netflixhistory_analysis_tv['media'] = 'TV'
        netflixhistory_analysis_tv['tmdb_id'] = netflixhistory_analysis_tv['tv_details'].apply(lambda x: x['id'])

        # drop irrelevant columns
        netflixhistory_analysis_tv = netflixhistory_analysis_tv.drop(columns=['tv_details', 'id','title_api','Title','Date'])


    # check if netflixhistory_analysis_movie is not empty
    if not netflixhistory_analysis_movie.empty:
        with st.spinner('Getting Netflix Movie Data'):
            # add column for movie details in netflixhistory_analysis_movie
            netflixhistory_analysis_movie['movie_details'] = netflixhistory_analysis_movie['title_api'].apply(getFilmDetailsNetflix)
            # drop rows with movie_details = Not Movie from netflixhistory_analysis_movie
            netflixhistory_analysis_movie = netflixhistory_analysis_movie[netflixhistory_analysis_movie['movie_details'] != 'Not Movie']
             # for Movie
            movieGenreList = getMovieGenreList()

            # add relevant column in netflixhistory_analysis_movie extracted from movie_details dict value column
            netflixhistory_analysis_movie['id'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['id'])
            netflixhistory_analysis_movie['IMDb_ID'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: getFilmIMDbID(str(x['id'])))
            netflixhistory_analysis_movie['adult'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['adult'])
            netflixhistory_analysis_movie['title'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['title'])
            netflixhistory_analysis_movie['original_language'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['original_language'])
            netflixhistory_analysis_movie['overview'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['overview'].replace('\n', ' '))
            netflixhistory_analysis_movie['vote_average'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['vote_average'])
            netflixhistory_analysis_movie['vote_count'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['vote_count'])
            netflixhistory_analysis_movie['popularity'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['popularity'])
            netflixhistory_analysis_movie['release_date'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['release_date'])
            netflixhistory_analysis_movie['genres'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: getGenresFromID(movieGenreList, x['genre_ids']))
            netflixhistory_analysis_movie['poster_path'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['poster_path'])
            netflixhistory_analysis_movie['media'] = 'movie'
            netflixhistory_analysis_movie['tmdb_id'] = netflixhistory_analysis_movie['movie_details'].apply(lambda x: x['id'])

            # drop irrelevant columns
            netflixhistory_analysis_movie = netflixhistory_analysis_movie.drop(columns=['movie_details','id', 'title_api','Title','Date','tv_details'])


        # combine netflixhistory_analysis_tv and netflixhistory_analysis_movie to find genre
        df_all = pd.concat([netflixhistory_analysis_tv, netflixhistory_analysis_movie], ignore_index=True)
    else:
        df_all = netflixhistory_analysis_tv

    # drop rows with no genres
    df_all = df_all[df_all['genres'].notna()]
    top_genres, least_genre = get_genres(df_all)

    if not netflixhistory_analysis_movie.empty:
        # apply rating rules for movies
        netflixhistory_analysis_movie = netflixhistory_analysis_movie[netflixhistory_analysis_movie['vote_average'].notna()]
        netflixhistory_analysis_movie = applyMovieRating(netflixhistory_analysis_movie, top_genres, least_genre)
        # apply rating rules for tv
        netflixhistory_analysis_tv = netflixhistory_analysis_tv[netflixhistory_analysis_tv['vote_average'].notna()]
        netflixhistory_analysis_tv = applyTVRating(netflixhistory_analysis_tv, top_genres, least_genre)

        # combine netflixhistory_analysis_tv and netflixhistory_analysis_movie using concat
        df_netflix_metadata = pd.concat([netflixhistory_analysis_tv, netflixhistory_analysis_movie], ignore_index=True)
    else:
        # apply rating rules for tv
        netflixhistory_analysis_tv = netflixhistory_analysis_tv[netflixhistory_analysis_tv['vote_average'].notna()]
        netflixhistory_analysis_tv = applyTVRating(netflixhistory_analysis_tv, top_genres, least_genre)
        df_netflix_metadata = netflixhistory_analysis_tv

    # drop frequency column
    df_netflix_metadata = df_netflix_metadata.drop(columns=['frequency'])
    # add columns for ['letterboxd_id', 'liked', 'letterboxd_link']
    df_netflix_metadata['letterboxd_id'] = ""
    df_netflix_metadata['liked'] = ""
    df_netflix_metadata['letterboxd_link'] = ""
    return df_netflix_metadata