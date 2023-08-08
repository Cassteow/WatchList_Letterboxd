import pandas as pd
import json
from individual_reco import getPyMongoDB

# Check input type
def check_input_type(username, netflixhistory):
    # Check input type
    if username == "" and netflixhistory is None:
        input_type = "Error"
    elif username == "" and netflixhistory is not None:
        input_type = "netflix"
    elif username != "" and netflixhistory is None:
        input_type = "username"
    else:
        input_type = "both"
    return input_type

# Function to clean movie df
def clean_movie_df(df):

    # drop column _id
    df = df.drop(['_id'], axis=1)
    # remove duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    # drop rows with no imdb_id
    df = df.dropna(subset=['imdb_id'])
    # drop rows with no title
    df = df.dropna(subset=['title'])

    # combine keywords
    df_keywords = getPyMongoDB("keywords")
    df_keywords = df_keywords.drop(['_id'], axis=1)
    df = pd.merge(df, df_keywords, on='id', how = 'left')
    # Filling the numm values as []
    df['keywords'].fillna('[]', inplace=True)

    # Fetching the keyword list from the column     
    df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in eval(x)])

    # Remove the expty spaces and join all the keyword with spaces
    df['keywords'] = df['keywords'].apply(lambda x: ' '.join([i.replace(" ",'') for i in x]))

    return df

# Function to filter df by year and rating
def filter_year_and_rating(df, year_range, rating_range):
    # convert release date to datetime
    df['release_date'] = pd.to_datetime(df['release_date'])

    # add year column
    df['year'] = df['release_date'].dt.year
    
    df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    df = df[(df['vote_average'] >= rating_range[0]) & (df['vote_average'] <= rating_range[1])]
    return df

# Function to extract genre keywords only from given row
def extract_genre(row):
    return [item['name'] for item in row]

# Function to extract genre keywords only from given row
def extract_json_to_list(row):
    # replace single quotes with double quotes
    row = row.replace("'", '"')
    json_row = json.loads(row)
    return [item['name'] for item in json_row]


# Function to check genres by row for filtering
def filter_genres_row(row, genres_to_filter):
    movie_genres = row['genres_list']
    for genre in movie_genres:
        if genre not in genres_to_filter:
            return False
    return True

# Function to filter genres
def filter_genres(df, genres_to_filter):
    # extract genre keywords only into a list
    df['genres_list'] = df['genres'].apply(extract_json_to_list)
    # filter out movies that do not contain any of the genres to filter
    df = df[df.apply(lambda row: filter_genres_row(row, genres_to_filter), axis=1)]
    return df