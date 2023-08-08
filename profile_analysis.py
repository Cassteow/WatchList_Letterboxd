import pandas as pd
import nltk

# Get Most Watched Genre
def get_most_watched_genre(df_user):
    genres = []
    for index, row in df_user.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres.append(genre)
    genres = ';'.join(genres) # convert list of lists to list of strings
    genres = genres.split(";")
    genres = nltk.FreqDist(genres)
    most_watched_genre = genres.most_common(1)[0][0]
    second_most_watched_genre = genres.most_common(2)[1][0]
    most_watched_genre = most_watched_genre + " and " + second_most_watched_genre
    return most_watched_genre

# Get Least Watched Genre
def get_least_watched_genre(df_user):
    genres = []
    for index, row in df_user.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres.append(genre)
    genres = ';'.join(genres) # convert list of lists to list of strings
    genres = genres.split(";")
    genres = nltk.FreqDist(genres)
    least_watched_genre = genres.most_common()[-1][0]
    return least_watched_genre

# Get Number of Genres Watched
def get_num_genres_watched(df_user):
    genres = []
    for index, row in df_user.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres.append(genre)
    genres = ';'.join(genres) # convert list of lists to list of strings
    genres = genres.split(";")
    genres = nltk.FreqDist(genres)
    num_genres_watched = len(genres)
    return num_genres_watched

# Get Most Popular Movie
def get_most_popular_movie(df_user):
    df_user['popularity'] = pd.to_numeric(df_user['popularity'], errors='coerce')
    most_popular_movie = df_user.sort_values(by=['popularity'], ascending=False)
    most_popular_movie = most_popular_movie.iloc[0]
    return most_popular_movie

# Get Least Popular Movie
def get_least_popular_movie(df_user):
    df_user['popularity'] = pd.to_numeric(df_user['popularity'], errors='coerce')
    least_popular_movie = df_user.loc[df_user['popularity'].idxmin()]
    return least_popular_movie

# Get Decade with Most Movies Watched
def get_decade_most_movies_watched(df_user):
    # get frequent decade of movies released
    decade_1980s = 0
    decade_1990s = 0
    decade_2000s = 0
    decade_2010s = 0
    decade_2020s = 0
    for index, row in df_user.iterrows():
        if row['release_date'] != '':
            year = int(row['release_date'].split("-")[0])
            if year < 1990:
                decade_1980s += 1
            elif year < 2000:
                decade_1990s += 1
            elif year < 2010:
                decade_2000s += 1
            elif year < 2020:
                decade_2010s += 1
            else:
                decade_2020s += 1
    # get decade with most movies watched
    decades = [decade_1980s, decade_1990s, decade_2000s, decade_2010s, decade_2020s]
    decade = decades.index(max(decades))
    if decade == 0:
        decade = "1980s and earlier"
    elif decade == 1:
        decade = "1990s"
    elif decade == 2:
        decade = "2000s"
    elif decade == 3:
        decade = "2010s"
    else:
        decade = "2020s"
    return decade

# Get top 5 genres of movies 
def get_top5_genres(df_user):
    df_get_top_genres = df_user.copy()
    genres = []
    # convert genres column to string
    df_get_top_genres['genres'] = df_get_top_genres['genres'].astype(str)
    for index, row in df_get_top_genres.iterrows():
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

# Get top 10 genres of movies 
def get_top10_genres(df_user):
    df_get_top_genres = df_user.copy()
    genres = []
    # convert genres column to string
    df_get_top_genres['genres'] = df_get_top_genres['genres'].astype(str)
    for index, row in df_get_top_genres.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres.append(genre)
    genres = ';'.join(genres) # convert list of lists to list of strings
    genres = genres.split(";")
    genres = nltk.FreqDist(genres)
    top_genres = genres.most_common(10)
    top_genres = [genre[0] for genre in top_genres]
    return top_genres

# ========= Group Analysis (2 users) =========

# Get Genre match between 2 users
def get_genre_match_score(df_user1, df_user2):
    # Get top 5 genres for both users
    user1_top_genres = get_top5_genres(df_user1)
    user2_top_genres = get_top5_genres(df_user2)
    # Check number matches in top 5 genres for both users
    num_matches = 0
    for genre in user1_top_genres:
        if genre in user2_top_genres:
            num_matches += 1 
    return num_matches*10 # times 10 for the total 50% score

# Get Movie match between 2 users
def get_movie_match_score(df_user1_full, df_user2_full):
    df_get_num_same_movies = df_user1_full.copy()
    df_get_num_same_movies = df_get_num_same_movies[df_get_num_same_movies['title'].isin(df_user2_full['title'])]
    score = len(df_get_num_same_movies)
    return score

# Get Genre match between 2 users
def get_fav_genre(df_user1_ori, df_user2_ori):
    df_user1 = df_user1_ori.copy()
    df_user2 = df_user2_ori.copy()
    # arrange all genres in df_user1 according to frequency
    genres_user1 = []
    # convert genres column to string
    df_user1['genres'] = df_user1['genres'].astype(str)
    for index, row in df_user1.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres_user1.append(genre)
    genres_user1 = ';'.join(genres_user1) # convert list of lists to list of strings
    genres_user1 = genres_user1.split(";")
    genres_user1 = nltk.FreqDist(genres_user1)
    genres_user1 = genres_user1.most_common()
    # arrange all genres in df_user2 according to frequency
    genres_user2 = []
    # convert genres column to string
    df_user2['genres'] = df_user2['genres'].astype(str)
    for index, row in df_user2.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres_user2.append(genre)
    genres_user2 = ';'.join(genres_user2) # convert list of lists to list of strings
    genres_user2 = genres_user2.split(";")
    genres_user2 = nltk.FreqDist(genres_user2)
    genres_user2 = genres_user2.most_common()
    # get first match in both lists
    for genre1 in genres_user1:
        for genre2 in genres_user2:
            if genre1[0] == genre2[0]:
                return genre1[0]
    return "None"

# Get one movie that both users have watched with highest rating and popularity
def get_common_movie(df_user1, df_user2):
    # get list of movies watched by both users
    df_user1_common = df_user1[df_user1['title'].isin(df_user2['title'])]
    # check if there are any movies in common
    if df_user1_common.empty:
        return "None"
    else:
        # sort by popularity
        df_user1_common = df_user1_common.sort_values(by=['rating','liked'], ascending=[False, False])
        # get top movie
        top_movie = df_user1.iloc[0]['title']
        return top_movie
    
# ========= Group Analysis (3 users) =========

# Get Genre match between 3 users
def get_genre_match_score_3users(df_user1, df_user2, df_user3):
    # Get top 10 genres for all users
    user1_top_genres = get_top10_genres(df_user1)
    user2_top_genres = get_top10_genres(df_user2)
    user3_top_genres = get_top10_genres(df_user3)
    # Check number matches in top 5 genres for all users
    num_matches = 0
    for genre in user1_top_genres:
        if genre in user2_top_genres:
            if genre in user3_top_genres:
                num_matches += 1 
    score = num_matches*10 # times 10 for the total 50% score
    if score > 50:
        score = 50
    return score 

# Get Movie match between 3 users
def get_movie_match_score_3users(df_user1_full, df_user2_full, df_user3_full):
    df_get_num_same_movies = df_user1_full.copy()
    df_get_num_same_movies = df_get_num_same_movies[df_get_num_same_movies['title'].isin(df_user2_full['title'])]
    df_get_num_same_movies = df_get_num_same_movies[df_get_num_same_movies['title'].isin(df_user3_full['title'])]
    score = len(df_get_num_same_movies)
    return score

# Get Genre match between 3 users
def get_fav_genre_3users(df_user1_ori, df_user2_ori, df_user3_ori):
    df_user1 = df_user1_ori.copy()
    df_user2 = df_user2_ori.copy()
    df_user3 = df_user3_ori.copy()
    # arrange all genres in df_user1 according to frequency
    genres_user1 = []
    # convert genres column to string
    df_user1['genres'] = df_user1['genres'].astype(str)
    for index, row in df_user1.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres_user1.append(genre)
    genres_user1 = ';'.join(genres_user1) # convert list of lists to list of strings
    genres_user1 = genres_user1.split(";")
    genres_user1 = nltk.FreqDist(genres_user1)
    genres_user1 = genres_user1.most_common()
    # arrange all genres in df_user2 according to frequency
    genres_user2 = []
    # convert genres column to string
    df_user2['genres'] = df_user2['genres'].astype(str)
    for index, row in df_user2.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres_user2.append(genre)
    genres_user2 = ';'.join(genres_user2) # convert list of lists to list of strings
    genres_user2 = genres_user2.split(";")
    genres_user2 = nltk.FreqDist(genres_user2)
    genres_user2 = genres_user2.most_common()
    # arrange all genres in df_user3 according to frequency
    genres_user3 = []
    # convert genres column to string
    df_user3['genres'] = df_user3['genres'].astype(str)
    for index, row in df_user3.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres_user3.append(genre)
    genres_user3 = ';'.join(genres_user3) # convert list of lists to list of strings
    genres_user3 = genres_user3.split(";")
    genres_user3 = nltk.FreqDist(genres_user3)
    genres_user3 = genres_user3.most_common()
    # get first match in all lists
    for genre1 in genres_user1:
        for genre2 in genres_user2:
            for genre3 in genres_user3:
                if genre1[0] == genre2[0]:
                    if genre1[0] == genre3[0]:
                        return genre1[0]
    return "None"

# Get one movie that all users have watched with highest rating and popularity
def get_common_movie_3users(df_user1, df_user2, df_user3):
    # get list of movies watched by all users
    df_user1_common = df_user1[df_user1['title'].isin(df_user2['title'])]
    df_user1_common = df_user1_common[df_user1_common['title'].isin(df_user3['title'])]
    # check if there are any movies in common
    if df_user1_common.empty:
        return "None"
    else:
        # sort by popularity
        df_user1_common = df_user1_common.sort_values(by=['rating','liked'], ascending=[False, False])
        # get top movie
        top_movie = df_user1.iloc[0]['title']
        return top_movie