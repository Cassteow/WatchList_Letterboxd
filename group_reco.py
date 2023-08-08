import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords


# Function to check who is expert user for 2 users
def check_expert_user(df_full_user1, df_full_user2):
    # count items
    count_user1 = df_full_user1['title'].count()
    count_user2 = df_full_user2['title'].count()
    # if one user has 70% of the total items, return the user as expert
    if count_user1 > count_user2:
        if count_user1/(count_user2+count_user1) >= 0.7:
            return "user1"
        else:
            return "no expert"
    elif count_user2 > count_user1:
        if count_user2/(count_user2+count_user1) >= 0.7:
            return "user2"
        else:
            return "no expert"
    else:
        return "no expert"
    
# Function to check who is expert user for 3 users
def check_expert_user_3users(df_full_user1, df_full_user2, df_full_user3):
    # count items
    count_user1 = df_full_user1['title'].count()
    count_user2 = df_full_user2['title'].count()
    count_user3 = df_full_user3['title'].count()

    # if one user has 50% of the total items, return the user as expert
    if count_user1 > count_user2 and count_user1 > count_user3:
        if count_user1/(count_user2 + count_user3+count_user1) >= 0.5:
            return "user1"
        else:
            return "no expert"
    elif count_user2 > count_user1 and count_user2 > count_user3:
        if count_user2/(count_user1 + count_user3+count_user2) >= 0.5:
            return "user2"
        else:
            return "no expert"
    elif count_user3 > count_user1 and count_user3 > count_user2:
        if count_user3/(count_user1 + count_user2+count_user3) >= 0.5:
            return "user3"
        else:
            return "no expert"
    else:
        return "no expert"
    
# Get SVD Results for 2 users
def getGroupSVDResults_2Users(df_SVD_Result_1, df_SVD_Result_2, expertUserCheck):
    # Rename SVDRatings column
    df_SVD_Result_1 = df_SVD_Result_1.rename(columns={'SVDRatings': 'SVDRatings_1'})
    df_SVD_Result_2 = df_SVD_Result_2.rename(columns={'SVDRatings': 'SVDRatings_2'})
    # add SVDRatings_2 to df_SVD_Result_1
    df_SVD_Result_1['SVDRatings_2'] = df_SVD_Result_2['SVDRatings_2']
    # copy df_SVD_Result_1 as df_SVD_Result_Group
    df_SVD_Result_Group = df_SVD_Result_1.copy()

    # calculate Group SVDRatings
    if expertUserCheck == "no expert" or expertUserCheck == None:
        # create a new column SVDRatings_Group to store the average of SVDRatings_1, SVDRatings_2
        df_SVD_Result_Group['SVDRatings_Group'] = df_SVD_Result_Group[['SVDRatings_1', 'SVDRatings_2']].mean(axis=1)
    elif expertUserCheck == "user1":
        # create a new column SVDRatings_Group that is 55% of SVDRatings_1 column and 45% of SVDRatings_2 column
        df_SVD_Result_Group['SVDRatings_Group'] = (df_SVD_Result_Group['SVDRatings_1'] * 0.55) + (df_SVD_Result_Group['SVDRatings_2'] * 0.45)
    elif expertUserCheck == "user2":
        # create a new column SVDRatings_Group to store the 55% of SVDRatings_2 and 45% of SVDRatings_1
        df_SVD_Result_Group['SVDRatings_Group'] = (df_SVD_Result_Group['SVDRatings_2'] * 0.55) + (df_SVD_Result_Group['SVDRatings_1'] * 0.45)
    # rearrange df_SVD_Result_Group columns based on SVDRatings_Group
    df_SVD_Result_Group = df_SVD_Result_Group.sort_values(by=['SVDRatings_Group'], ascending=False)
    return df_SVD_Result_Group

# Get top 5 genres of movies in df_user_letterboxd (for 2 users)
def get_group2_top_genres(df_letterboxd_1, df_letterboxd_2):
    # combine all sorted dfs
    df_group_letterboxd = pd.concat([df_letterboxd_1, df_letterboxd_2], ignore_index=True)
    genres = []
    for index, row in df_group_letterboxd.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres.append(genre)
    genres = ' '.join(genres) # convert list of lists to list of strings
    genres = genres.split(" ")
    genres = nltk.FreqDist(genres)
    top_genres = genres.most_common(5)
    top_genres = [genre[0] for genre in top_genres]
    return top_genres

# Function to get CBF input for 2 group user
def get_2Group_filtered_CBF_input(df_svd_results, sorted_df1, sorted_df2, top_genres):
    # Sort df_svd_results according to SVDRatings
    df_svd_results = df_svd_results.sort_values(by=['SVDRatings_Group','popularity','vote_average','vote_count'], ascending = [False, False, False, False])
    # Drop rows of movies that do not belong in top_genres
    df_svd_results = df_svd_results[df_svd_results['genres'].str.contains('|'.join(top_genres))]
    # Drop rows of movies with vote_count < 50
    df_svd_results = df_svd_results[df_svd_results['vote_count'] > 50]
    # Drop rows of movies with same imdb_id values in df_user_letterboxd
    df_svd_results = df_svd_results[~df_svd_results['imdb_id'].isin(sorted_df1['IMDb_ID'])]
    df_svd_results = df_svd_results[~df_svd_results['imdb_id'].isin(sorted_df2['IMDb_ID'])]
    # Return top 250 movies in df_svd_results
    df_svd_results = df_svd_results.head(250)
    return df_svd_results

# Get sorted group letterboxd df for 2 users
def get_sorted_2group_letterboxd_df(sorted_df_letterboxd_1, sorted_df_letterboxd_2, expertUserCheck):
    if expertUserCheck == "no expert" or expertUserCheck == None:
        # get top 5 of both dfs
        sorted_group_df = pd.concat([sorted_df_letterboxd_1.head(5), sorted_df_letterboxd_2.head(5)], ignore_index=True)
    elif expertUserCheck == "user1":
        # get top 6 for expert, top 4 for non-expert of both dfs
        sorted_group_df = sorted_df_letterboxd_1.head(6).append(sorted_df_letterboxd_2.head(4), ignore_index=True)
    elif expertUserCheck == "user2":
        # get top 6 for expert, top 4 for non-expert of both dfs
        sorted_group_df = pd.concat([sorted_df_letterboxd_2.head(6), sorted_df_letterboxd_1.head(4)], ignore_index=True)
    return sorted_group_df

# Get SVD Results for 3 users
def getGroupSVDResults_3Users(df_SVD_Result_1, df_SVD_Result_2, df_SVD_Result_3, expertUserCheck):
    # Rename SVDRatings column
    df_SVD_Result_1 = df_SVD_Result_1.rename(columns={'SVDRatings': 'SVDRatings_1'})
    df_SVD_Result_2 = df_SVD_Result_2.rename(columns={'SVDRatings': 'SVDRatings_2'})
    df_SVD_Result_3 = df_SVD_Result_3.rename(columns={'SVDRatings': 'SVDRatings_3'})
    # add SVDRatings_2 and SVDRatings_3 to df_SVD_Result_1
    df_SVD_Result_1['SVDRatings_2'] = df_SVD_Result_2['SVDRatings_2']
    df_SVD_Result_1['SVDRatings_3'] = df_SVD_Result_3['SVDRatings_3']

    # copy df_SVD_Result_1 as df_SVD_Result_Group
    df_SVD_Result_Group = df_SVD_Result_1.copy()

    # calculate Group SVDRatings
    if expertUserCheck == "no expert" or expertUserCheck == None:
        # create a new column SVDRatings_Group to store the average of SVDRatings_1, SVDRatings_2, SVDRatings_3
        df_SVD_Result_Group['SVDRatings_Group'] = df_SVD_Result_Group[['SVDRatings_1', 'SVDRatings_2', 'SVDRatings_3']].mean(axis=1)
        
    elif expertUserCheck == "user1":
        # create a new column SVDRatings_Group that is 40% of SVDRatings_1 column and 30% of SVDRatings_2 and SVDRatings_3 column
        df_SVD_Result_Group['SVDRatings_Group'] = (df_SVD_Result_Group['SVDRatings_1'] * 0.4) + (df_SVD_Result_Group['SVDRatings_2'] * 0.3) + (df_SVD_Result_Group['SVDRatings_3'] * 0.3)
    
    elif expertUserCheck == "user2":
        # create a new column SVDRatings_Group that is 40% of SVDRatings_2 column and 30% of SVDRatings_1 and SVDRatings_3 column
        df_SVD_Result_Group['SVDRatings_Group'] = (df_SVD_Result_Group['SVDRatings_2'] * 0.4) + (df_SVD_Result_Group['SVDRatings_1'] * 0.3) + (df_SVD_Result_Group['SVDRatings_3'] * 0.3)
    
    elif expertUserCheck == "user3":
        # create a new column SVDRatings_Group that is 40% of SVDRatings_3 column and 30% of SVDRatings_1 and SVDRatings_2 column
        df_SVD_Result_Group['SVDRatings_Group'] = (df_SVD_Result_Group['SVDRatings_3'] * 0.4) + (df_SVD_Result_Group['SVDRatings_1'] * 0.3) + (df_SVD_Result_Group['SVDRatings_2'] * 0.3)


    # rearrange df_SVD_Result_Group columns based on SVDRatings_Group
    df_SVD_Result_Group = df_SVD_Result_Group.sort_values(by=['SVDRatings_Group'], ascending=False)
    return df_SVD_Result_Group


# Get sorted group letterboxd df for 3 users
def get_sorted_3group_letterboxd_df(sorted_df_letterboxd_1, sorted_df_letterboxd_2, sorted_df_letterboxd_3, expertUserCheck):
    if expertUserCheck == "no expert" or expertUserCheck == None:
        # get top 5 of all dfs
        sorted_group_df = pd.concat([sorted_df_letterboxd_1.head(5), sorted_df_letterboxd_2.head(5)], ignore_index=True)
        sorted_group_df = pd.concat([sorted_group_df, sorted_df_letterboxd_3.head(5)], ignore_index=True)
    elif expertUserCheck == "user1":
        # get top 7 for expert, top 4 for non-expert of both dfs
        sorted_group_df = sorted_df_letterboxd_1.head(7).append(sorted_df_letterboxd_2.head(4), ignore_index=True)
        sorted_group_df = pd.concat([sorted_group_df, sorted_df_letterboxd_3.head(4)], ignore_index=True)
    elif expertUserCheck == "user2":
        # get top 7 for expert, top 4 for non-expert of both dfs
        sorted_group_df = pd.concat([sorted_df_letterboxd_2.head(7), sorted_df_letterboxd_1.head(4)], ignore_index=True)
        sorted_group_df = pd.concat([sorted_group_df, sorted_df_letterboxd_3.head(4)], ignore_index=True)
    elif expertUserCheck == "user3":
        # get top 7 for expert, top 4 for non-expert of both dfs
        sorted_group_df = pd.concat([sorted_df_letterboxd_3.head(7), sorted_df_letterboxd_1.head(4)], ignore_index=True)
        sorted_group_df = pd.concat([sorted_group_df, sorted_df_letterboxd_2.head(4)], ignore_index=True)
    return sorted_group_df

# Get top 5 genres of movies in df_user_letterboxd (for 3 users)
def get_group3_top_genres(df_letterboxd_1, df_letterboxd_2, df_letterboxd_3):
    # combine all sorted dfs
    df_group_letterboxd = pd.concat([df_letterboxd_1, df_letterboxd_2], ignore_index=True)
    df_group_letterboxd = pd.concat([df_group_letterboxd, df_letterboxd_3], ignore_index=True)
    genres = []
    for index, row in df_group_letterboxd.iterrows():
        if row['genres'] != '':
            list_of_genres = row['genres'].split(",")
            for genre in list_of_genres:
                genres.append(genre)
    genres = ' '.join(genres) # convert list of lists to list of strings
    genres = genres.split(" ")
    genres = nltk.FreqDist(genres)
    top_genres = genres.most_common(5)
    top_genres = [genre[0] for genre in top_genres]
    return top_genres

# Function to get CBF input
def get_3Group_filtered_CBF_input(df_svd_results, sorted_df1, sorted_df2, sorted_df3, top_genres):
    # Sort df_svd_results according to SVDRatings
    df_svd_results = df_svd_results.sort_values(by=['SVDRatings_Group','popularity','vote_average','vote_count'], ascending = [False, False, False, False])
    # Drop rows of movies that do not belong in top_genres
    df_svd_results = df_svd_results[df_svd_results['genres'].str.contains('|'.join(top_genres))]
    # Drop rows of movies with vote_count < 50
    df_svd_results = df_svd_results[df_svd_results['vote_count'] > 50]
    # Drop rows of movies with same imdb_id values in df_user_letterboxd
    df_svd_results = df_svd_results[~df_svd_results['imdb_id'].isin(sorted_df1['IMDb_ID'])]
    df_svd_results = df_svd_results[~df_svd_results['imdb_id'].isin(sorted_df2['IMDb_ID'])]
    df_svd_results = df_svd_results[~df_svd_results['imdb_id'].isin(sorted_df3['IMDb_ID'])]
    # Return top 1000 movies in df_svd_results
    df_svd_results = df_svd_results.head(250)
    return df_svd_results

# Function to get cosine similarity of movies in df_movies
def get_group_CBF_cosine_sim(df_movies, sorted_df):
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
    df_movies['finalRatings'] = (df_movies['cosine_sim'] * 30) + df_movies['SVDRatings_Group']
    # print max and min cosine_sim
    print("Max cosine_sim: ", df_movies['cosine_sim'].max())
    print("Min cosine_sim: ", df_movies['cosine_sim'].min())
    #df_movies['SVDRatings'] = df_movies['SVDRatings'] * 0.2
    #df_movies['cosine_sim'] = df_movies['cosine_sim'] + df_movies['SVDRatings']
    df_cbf_results = df_movies.sort_values(by='finalRatings', ascending = False)
    return df_cbf_results
