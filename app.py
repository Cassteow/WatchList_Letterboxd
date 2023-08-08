import streamlit as st
import pandas as pd
from datetime import date

from netflix_processing import process_netflixhistory, getNetflixMetadata, check_netflixhistory
from letterboxd_processing import scrape_all_films, scrape_films_one_page, getFilmMetadataDF, get_poster_path
from preprocessing import clean_movie_df, filter_year_and_rating, filter_genres, extract_json_to_list, check_input_type

from individual_reco import getPyMongoDB, getUserRatingSVDDF, trainSVDModel, getSVDRecommendations
from individual_reco import get_SVD_Dataframe, combineRatingsDF, getAverageRating
from individual_reco import sort_rated_df, get_top_genres, get_CBF_description, get_CBF_description_letterboxd
from individual_reco import get_filtered_CBF_input, get_CBF_cosine_sim

from group_reco import check_expert_user, check_expert_user_3users
from group_reco import getGroupSVDResults_2Users, getGroupSVDResults_3Users, get_group2_top_genres, get_group3_top_genres
from group_reco import get_sorted_2group_letterboxd_df, get_sorted_3group_letterboxd_df
from group_reco import get_2Group_filtered_CBF_input, get_3Group_filtered_CBF_input, get_group_CBF_cosine_sim

from profile_analysis import get_most_watched_genre, get_least_watched_genre, get_num_genres_watched
from profile_analysis import get_most_popular_movie, get_least_popular_movie, get_decade_most_movies_watched
from profile_analysis import get_genre_match_score, get_movie_match_score, get_fav_genre, get_common_movie
from profile_analysis import get_genre_match_score_3users, get_movie_match_score_3users, get_fav_genre_3users, get_common_movie_3users


st.set_page_config(page_icon="📽️",page_title="WatchList", layout="wide")

# ----- SIDEBAR -----
# Sidebar for recommendation options
st.sidebar.title("📝 Recommendation Type")
reco_type = st.sidebar.selectbox("Select recommendation type", ("Individual", "Group (2 users)", "Group (3 users)"))

# Individual Recommendation
if reco_type == "Individual":
    # ----- HEADER -----
    with st.container():
        st.title('🎬 WatchList')
        st.subheader("A Hybrid Recommendation System for Film and Television")
        st.markdown("Select **:green[individual 🧍]** or **:green[group 👪]** recommendation option from the side bar.")
        st.markdown("Provide WatchList with your **:blue[Letterboxd username],** or **:blue[ Netflix Viewing History]** (csv file), or **:blue[both]**!")
        with st.expander("More Information"):
            st.write("WatchList is a hybrid recommendation system that combines **:green[collaborative filtering]** and **:green[content-based filtering]** to provide you with movie and TV recommendations.")
            st.write("")
            st.markdown("You may choose to provide WatchList with your **:blue[Letterboxd username]**, **:blue[ Netflix Viewing History]** (csv file), or both.")
            st.write("")
            st.markdown("If you choose to provide your **:blue[Letterboxd username]**, WatchList will scrape your profile and ratings from Letterboxd.")
            st.write("")
            st.markdown("If you choose to provide your **:blue[ Netflix Viewing History]**, WatchList will process your viewing history and scrape the metadata of the movies and TV shows you have watched.")
            st.write("Due to the limitations of Netflix Viewing History, WatchList is only able to process your viewing history"+
                     " if you have watched at least 5 titles using your Netflix account, and may be **:red[inaccurate]** in gathering the correct titles.")
            st.write("")
            st.markdown("**:red[Disclaimer:]** WatchList does not store your Letterboxd username, Netflix Viewing History, or any user data. WatchList is not affiliated with Letterboxd or Netflix.")
            st.markdown("**It will take some time for WatchList to scrape your Letterboxd profile and process your Netflix Viewing History. Please be patient!**")
            st.write("")
        # empty line
        st.write("")
        st.write("")

    # ----- INPUT -----
    with st.container():
        # create two columns for input
        row_input = st.columns((2,1,2,1))
        # username input at column 1
        with row_input[0]:
            # username input
            username = st.text_input('Letterboxd Username') 
            
        with row_input[2]:
            # netflixhistory input
            netflixhistory = st.file_uploader('Netflix Viewing History', type=['csv'],  accept_multiple_files=False, 
                                              help="Download your Netflix Viewing History in csv file format from your account settings.")
            
        # place filtering options in a container with smaller width
        with st.container():
            st.write("")
            st.write("")
            # create two columns for filtering options
            row_filter = st.columns((3,1,3,1))
            
            with row_filter[0]:
                # both way slider to filter movies by year
                year_range_filter = st.slider('Filter by year', 1900, 2023, (1900, 2023))
                # both way slider to filter movies by rating
                rating_filter = st.slider('Filter by average rating', min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)
            with row_filter[2]:
                # filter movies by genre
                genre_filter = st.multiselect('Filter by genre', ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 
                                                                        'Action', 'Crime', 'Thriller', 'Horror', 'History', 
                                                                        'Science Fiction', 'Mystery', 'War', 
                                                                        'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie'], 
                                                                default=['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 
                                                                        'Action', 'Crime', 'Thriller', 'Horror', 'History', 
                                                                        'Science Fiction', 'Mystery', 'War', 
                                                                        'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie'])
        #netflixhistory = st.file_uploader()
        row_button = st.columns((3,2,2,3)) 
        # submit and reset button in green colour
        submit = row_button[1].button('Submit') 
        reset = row_button[2].button('Reset')
        result = False

    if submit:
        result = True

    if reset:
        # reset all inputs to default
        result = False


    if result:
        # Get Movie Database as Dataframe
        df_moviesDB = getPyMongoDB("movies_metadata")
        # Clean movie df
        df_moviesDB = clean_movie_df(df_moviesDB)
        today = date.today()
        filename = "{0}_{1}".format(str(today), username)
        # Four types of Input: Letterboxd Username Only, Netflix History Only, Both, or None (ERROR)
        # Check input type
        input_type = check_input_type(username, netflixhistory)

        # check genre filter
        if len(genre_filter) == 0:
            st.error("Error. Genre filter cannot be left empty.")
            st.stop()
        

        # Check if input_type is Error
        if input_type == "Error":
            st.error("Error. Please input Letterboxd username or Netflix Viewing History.")
            st.stop()
        
        # Process netflix input
        if input_type == "netflix" or input_type == "both":
            # convert netflixhistory to dataframe
            netflixhistory = pd.read_csv(netflixhistory)

            # check validity of netflixhistory
            validity = check_netflixhistory(netflixhistory)
            if validity == "invalid":
                st.error("Error. Netflix Viewing History is invalid. Please try again.")
                st.stop()
            elif validity == "insufficient":
                st.error("Error. Netflix Viewing History is insufficient. Please make sure you have viewed at least 5 titles using your Netflix account.")
                st.stop()
            elif validity == "valid":
                # process netflixhistory
                df_netflixhistory = process_netflixhistory(netflixhistory)
                # get netflix metadata
                df_netflixhistory_metadata = getNetflixMetadata(df_netflixhistory)
                

        # Process Letterboxd input
        if input_type == "username" or input_type == "both":
            # Start scraping user profile
            with st.spinner('Scraping your movies on Letterboxd'):
                st.write("---")
                df_user_profile = scrape_films_one_page(username)
                # check if df_user_profile is empty
                if df_user_profile.empty:
                    st.error("Error. Username or profile ratings not found. Please try again.")
                    st.stop()
            with st.spinner('Getting movie details'):    
                df_user_profile_metadata = getFilmMetadataDF(df_user_profile)
        
        # Check if input_type is both
        if input_type == "both":
            # drop rows in df_netflixhistory_metadata that exists in df_user_profile_metadata
            df_netflixhistory_metadata = df_netflixhistory_metadata[~df_netflixhistory_metadata['IMDb_ID'].isin(df_user_profile_metadata['IMDb_ID'])]
            # Combine netflix metadata with letterboxd metadata
            df_user_profile_metadata = pd.concat([df_user_profile_metadata, df_netflixhistory_metadata], ignore_index=True)
        
        if input_type == "netflix":
            df_user_profile_metadata = df_netflixhistory_metadata
            df_user_profile = df_netflixhistory_metadata

        # Get profile analysis
        most_watched_genre = get_most_watched_genre(df_user_profile_metadata)
        least_watched_genre = get_least_watched_genre(df_user_profile_metadata)
        num_genres_watched = get_num_genres_watched(df_user_profile_metadata)
        most_popular_movie = get_most_popular_movie(df_user_profile_metadata)
        least_popular_movie = get_least_popular_movie(df_user_profile_metadata)
        decade_most_movies_watched = get_decade_most_movies_watched(df_user_profile_metadata)
            
        #end1 = time.time()
        #print("Scraping time: ", end1-start)

            
        with st.spinner('Getting recommendations for you'):
            
            # Filter movies based on user input
            df_moviesDB = filter_year_and_rating(df_moviesDB, year_range_filter, rating_filter)
            # Filter movies based on genres
            df_moviesDB = filter_genres(df_moviesDB, genre_filter)
            
            # Collaborative Filtering - SVD
            # Get User Rating Input DF for SVD
            df_userRatingSVD = getUserRatingSVDDF(df_movies_db=df_moviesDB, df_user_initial=df_user_profile, username=9999999)

            # check if df_userRatingSVD is empty
            if df_userRatingSVD.empty:
                # add a column SVDRatings as 0 as df_SVD_Results
                df_SVD_Results = df_moviesDB.copy()
                df_SVD_Results['SVDRatings'] = 0
            else:
                # Get Average Rating of User
                averageRating = getAverageRating(df_userRatingSVD)
                # Combine user ratings with rating df
                df_userRatingSVD = combineRatingsDF(df_userRatingSVD)
                # Train SVD model
                svd_model = trainSVDModel(df_userRatingSVD)
                # Get List of Sorted SVD Recommendations
                list_SVD_reco = getSVDRecommendations(data=df_userRatingSVD, df_movies=df_moviesDB, user_id=9999999, algo=svd_model)
            
                # Get Dataframe for SVD Recommendations
                df_SVD_Results = get_SVD_Dataframe(sorted_recommendations=list_SVD_reco, df_movies=df_moviesDB, user_id=9999999, avgRating=averageRating)
            
            # Content Based Filtering - CBF
            # Get top genres of user
            top_genres = get_top_genres(df_user_profile_metadata)
            # Get sorted df_user_letterboxd_metadata
            sorted_df_letterboxd = sort_rated_df(df_user_profile_metadata)
            # Get CBF description of movies
            df_SVD_Results = get_CBF_description(df_SVD_Results)
            sorted_df_letterboxd = get_CBF_description_letterboxd(sorted_df_letterboxd)
            # Get CBF Input
            df_CBF_input = get_filtered_CBF_input(df_svd_results=df_SVD_Results, sorted_df=sorted_df_letterboxd, top_genres=top_genres)     
            # Get CBF output with cosine similarity
            df_CBF_results = get_CBF_cosine_sim(df_movies=df_CBF_input, sorted_df=sorted_df_letterboxd)
            # remove duplicates
            df_CBF_results = df_CBF_results.drop_duplicates(subset=['imdb_id'])
            # Get top 20 CBF recommendations
            df_CBF_results = df_CBF_results.head(20)  
            
        # Reformat columns
        df_CBF_results['languages_list'] = df_CBF_results['spoken_languages'].apply(extract_json_to_list)
        st.header("🗒️ Your Individual Recommendation")
        row_movies = {}
        # if cbf results is empty
        if df_CBF_results.empty:
            st.error("No recommendations found. You may try again by reducing the restriction in your filtering option.")
            st.stop()
        for i, movie in enumerate(df_CBF_results['title']):
            if (i>=10):
                break
            row = int(i/2)
            if (i%2!=1): 
                row_movies[row] = st.columns(2)
            col = i%2
            # Print details of recommended movie
            url = "https://www.imdb.com/title/" + df_CBF_results['imdb_id'].values[i] +"/"
            row_movies[row][col].subheader("[{0}]({1})".format(movie, url))
            print_year = df_CBF_results['year'].values[i]
            print_year = "{:.0f}".format(print_year)
            row_movies[row][col].write("⭐Average Rating: {0}".format(df_CBF_results['vote_average'].values[i])+ " | 📅Release Year: {0}".format(print_year))
            
            print_genre = ""
            for genre in df_CBF_results['genres_list'].values[i]:
                print_genre += genre + ", "
            print_genre = print_genre[:-2]
            row_movies[row][col].write("🎭Genres: {0}".format(print_genre))
            print_lang = ""
            for lang in df_CBF_results['languages_list'].values[i]:
                print_lang += lang + ", "
            print_lang = print_lang[:-2]
            row_movies[row][col].write("🗣️Languages: {0}".format(print_lang))

            # Display movie poster
            poster_path = get_poster_path(df_CBF_results['imdb_id'].values[i])
            image_url = "https://image.tmdb.org/t/p/original"+poster_path
            

        with st.expander("See full list of recommendations"):
            
            # Drop unnecessary columns in df_CBF_results
            df_CBF_results = df_CBF_results[['title', 'year', 'vote_average', 'genres_list', 'languages_list', 'imdb_id']]
            df_CBF_results = df_CBF_results.reset_index(drop=True)
            st.write("")
            st.dataframe(df_CBF_results)
            def convert_df(df_recom):
                return df_recom.to_csv(index=False).encode('utf-8')
            csv = convert_df(df_CBF_results)
            st.download_button(
                "Download Recommenations",
                csv,
                "{}_Movie Recommendations.csv".format(filename),
                "text/csv",
                key='download-csv'
            )
            st.write("")
        st.write("---")
        st.header("📊 Your Profile Analysis")
        
        profile_analysis_row = st.columns((1,4,1,4,1,4,1))
        st.write("")
        st.subheader("🎭 Genres")
        if num_genres_watched > 16:
            st.subheader(":blue[The Expert: ]")
            st.markdown("With a vast repertoire of **:green[**{0}** ]** genres of movie and TV content under your belt, you have proven yourself to be an avid explorer. Way to go!".format(num_genres_watched))
        elif num_genres_watched > 12:
            st.subheader(":blue[The Connoisseur: ]")
            st.markdown("Your willingness to step out of your comfort zone and actively seek out films and shows from **:green[**{0}** ]** types of genres has allowed you to discover hidden gems!".format(num_genres_watched))
        elif num_genres_watched > 8:
            st.subheader(":blue[The Explorer: ]")
            st.markdown("You have watched movies and shows from **:green[**{0}** ]** genres. Keep exploring and you might find your favourite genre!".format(num_genres_watched))
        else:
            st.subheader(":blue[The Novice: ]")
            st.markdown("You have watched movies and shows from **:green[**{0}** ]** genres. Your openness to discovering new genres is commendable, and you have the exciting opportunity to broaden your movie and TV title repertoire.".format(num_genres_watched))
        st.markdown("It seems like you are a fan of **:green[**{0}** ]** movies or shows!".format(most_watched_genre))
        st.markdown("And uhh... not a fan of the **:green[**{0}** ]** genre huh :(".format(least_watched_genre))
        st.write("")    
        st.subheader("📈 Popularity")
        st.markdown("**:green[**{0}** ]** is the most popular movie/show you have watched!".format(most_popular_movie['title']))
        st.markdown("While your most obscure title is **:green[**{0}** ]**!".format(least_popular_movie['title']))
        st.write("")
        st.subheader("📅 Decade")
        st.markdown("You have watched the most content from the **:green[**{0}** ]**!".format(decade_most_movies_watched))

# Group Recommendation (2 users)
elif reco_type == "Group (2 users)":
    # ----- HEADER -----
    with st.container():
        st.title('🎬 WatchList')
        st.subheader("A Hybrid Recommendation System for Film and Television")
        st.markdown("Select **:green[individual 🧍]** or **:green[group 👪]** recommendation option from the side bar.")
        st.markdown("Provide WatchList with your **:blue[Letterboxd username],** or **:blue[ Netflix Viewing History]** (csv file), or **:blue[both]**!")
        with st.expander("More Information"):
            st.write("WatchList is a hybrid recommendation system that combines **:green[collaborative filtering]** and **:green[content-based filtering]** to provide you with movie and TV recommendations.")
            st.write("")
            st.markdown("You may choose to provide WatchList with your **:blue[Letterboxd username]**, **:blue[ Netflix Viewing History]** (csv file), or both.")
            st.write("")
            st.markdown("If you choose to provide your **:blue[Letterboxd username]**, WatchList will scrape your profile and ratings from Letterboxd.")
            st.write("")
            st.markdown("If you choose to provide your **:blue[ Netflix Viewing History]**, WatchList will process your viewing history and scrape the metadata of the movies and TV shows you have watched.")
            st.write("Due to the limitations of Netflix Viewing History, WatchList is only able to process your viewing history"+
                     " if you have watched at least 5 titles using your Netflix account, and may be **:red[inaccurate]** in gathering the correct titles.")
            st.write("")
            st.markdown("**:red[Disclaimer:]** WatchList does not store your Letterboxd username, Netflix Viewing History, or any user data. WatchList is not affiliated with Letterboxd or Netflix.")
            st.markdown("**It will take some time for WatchList to scrape your Letterboxd profile and process your Netflix Viewing History. Please be patient!**")
            st.write("")
        # empty line
        st.write("")
        st.write("")

    # ----- INPUT -----
    with st.container():
        # create two columns for input
        row_input = st.columns((2,1,3,1))
        # username input at column 1
        with row_input[0]:
            # username input
            username_1 = st.text_input('First User\'s Letterboxd Username')
            st.write("")
            st.write("") 
            username_2 = st.text_input('Second User\'s Letterboxd Username') 
        with row_input[2]:
            # netflixhistory input
            netflixhistory_1 = st.file_uploader('First User\'s Netflix Viewing History', type=['csv'], accept_multiple_files=False, 
                                              help="Download your Netflix Viewing History in csv file format from your account settings.")
            netflixhistory_2 = st.file_uploader('Second User\'s Netflix Viewing History', type=['csv'], accept_multiple_files=False, 
                                              help="Download your Netflix Viewing History in csv file format from your account settings.")
            
        # place filtering options in a container with smaller width
        st.write("")
        st.write("")
        st.write("")

        with st.container():
            # create two columns for filtering options
            row_filter = st.columns((3,0.7,4,1))
            
            with row_filter[0]:
                # both way slider to filter movies by year
                year_range_filter = st.slider('Filter by year', 1900, 2023, (1900, 2023))
                # both way slider to filter movies by rating
                rating_filter = st.slider('Filter by average rating', min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)
            with row_filter[2]:
                # filter movies by genre
                genre_filter = st.multiselect('Filter by genre', ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 
                                                                        'Action', 'Crime', 'Thriller', 'Horror', 'History', 
                                                                        'Science Fiction', 'Mystery', 'War', 
                                                                        'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie'], 
                                                                default=['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 
                                                                        'Action', 'Crime', 'Thriller', 'Horror', 'History', 
                                                                        'Science Fiction', 'Mystery', 'War', 
                                                                        'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie'])
        
        st.write("")
        st.write("")
        #netflixhistory = st.file_uploader()
        row_button = st.columns((4,2,2,4)) 
        # submit and reset button in green colour
        submit = row_button[1].button('Submit') 
        reset = row_button[2].button('Reset')
        result = False

    if submit:
        result = True

    if reset:
        # reset all inputs to default
        result = False

    if result:
        st.write("---")
        # Get Movie Database as Dataframe
        df_moviesDB = getPyMongoDB("movies_metadata")
        # Clean movie df
        df_moviesDB = clean_movie_df(df_moviesDB)
        today = date.today()
        filename = "{0}_{1}".format(str(today), username_1+"_"+username_2)

        # Check input type
        input_type_1 = check_input_type(username_1, netflixhistory_1)
        input_type_2 = check_input_type(username_2, netflixhistory_2)
        
        # check genre filter
        if len(genre_filter) == 0:
            st.error("Error. Genre filter cannot be left empty.")
            st.stop()
        
        # Check if input_type is Error
        if input_type_1 == "Error" or input_type_2 == "Error":
            st.error("Error. Please input Letterboxd username or Netflix Viewing History.")
            st.stop()
        
        # Process netflix input
        if input_type_1 == "netflix" or input_type_1 == "both":
            # convert netflixhistory to dataframe
            netflixhistory_1 = pd.read_csv(netflixhistory_1)
            # check validity of netflixhistory_1
            validity = check_netflixhistory(netflixhistory_1)
            if validity == "invalid":
                st.error("Error. Netflix Viewing History is invalid. Please try again.")
                st.stop()
            elif validity == "insufficient":
                st.error("Error. Netflix Viewing History is insufficient. Please make sure you have viewed at least 5 titles using your Netflix account.")
                st.stop()
            elif validity == "valid":
                # process netflixhistory
                df_netflixhistory_1 = process_netflixhistory(netflixhistory_1)
                # get netflix metadata
                df_netflixhistory_metadata_1 = getNetflixMetadata(df_netflixhistory_1)
        
        if input_type_2 == "netflix" or input_type_2 == "both":
            # convert netflixhistory to dataframe
            netflixhistory_2 = pd.read_csv(netflixhistory_2)
            # check validity of netflixhistory_2
            validity = check_netflixhistory(netflixhistory_2)
            if validity == "invalid":
                st.error("Error. Netflix Viewing History is invalid. Please try again.")
                st.stop()
            elif validity == "insufficient":
                st.error("Error. Netflix Viewing History is insufficient. Please make sure you have viewed at least 5 titles using your Netflix account.")
                st.stop()
            elif validity == "valid":
                # process netflixhistory
                df_netflixhistory_2 = process_netflixhistory(netflixhistory_2)
                # get netflix metadata
                df_netflixhistory_metadata_2 = getNetflixMetadata(df_netflixhistory_2)
        
        # Process Letterboxd input
        if input_type_1 == "username" or input_type_1 == "both":
            # Start scraping user profile
            with st.spinner('Scraping '+ username_1+'\'s movies on Letterboxd'):
                #start = time.time()
                df_user1_profile = scrape_films_one_page(username_1)
                # check if df_user_profile is empty
                if df_user1_profile.empty:
                    st.error("Error. Username or profile ratings not found. Please try again.")
                    st.stop()
                
                # get full profile
                df_full_user1_profile = scrape_all_films(username_1)
            with st.spinner('Getting movie details'):
                # Get film metadata through TMDB API    
                df_user1_profile_metadata = getFilmMetadataDF(df_user1_profile)
                
        if input_type_2 == "username" or input_type_2 == "both":
            with st.spinner('Scraping '+ username_2+'\'s movies on Letterboxd'):
                df_user2_profile = scrape_films_one_page(username_2)
                # check if df_user_profile is empty
                if df_user2_profile.empty:
                    st.error("Error. Username or profile ratings not found. Please try again.")
                    st.stop()
                
                # get full profile
                df_full_user2_profile = scrape_all_films(username_2)
            
            with st.spinner('Getting movie details'):
                # Get film metadata through TMDB API    
                df_user2_profile_metadata = getFilmMetadataDF(df_user2_profile)
        
        # Check if input_type is both
        if input_type_1 == "both":
            # drop rows in df_netflixhistory_metadata that exists in df_user_profile_metadata
            df_netflixhistory_metadata_1 = df_netflixhistory_metadata_1[~df_netflixhistory_metadata_1['IMDb_ID'].isin(df_user1_profile_metadata['IMDb_ID'])]
            # Combine netflix metadata with letterboxd metadata
            df_user1_profile_metadata = pd.concat([df_user1_profile_metadata, df_netflixhistory_metadata_1], ignore_index=True)

            # drop rows in df_netflixhistory_metadata that exists in df_full_user1_profile
            df_netflix_full_1 = df_netflixhistory_metadata_1[~df_netflixhistory_metadata_1['title'].isin(df_full_user1_profile['title'])]
            df_full_user1_profile = pd.concat([df_full_user1_profile, df_netflix_full_1], ignore_index=True)
            # reset index
            df_full_user1_profile = df_full_user1_profile.reset_index(drop=True)


        if input_type_2 == "both":
            # drop rows in df_netflixhistory_metadata that exists in df_user_profile_metadata
            df_netflixhistory_metadata_2 = df_netflixhistory_metadata_2[~df_netflixhistory_metadata_2['IMDb_ID'].isin(df_user2_profile_metadata['IMDb_ID'])]
            # Combine netflix metadata with letterboxd metadata
            df_user2_profile_metadata = pd.concat([df_user2_profile_metadata, df_netflixhistory_metadata_2], ignore_index=True)

            # drop rows in df_netflixhistory_metadata that exists in df_full_user2_profile
            df_netflix_full_2 = df_netflixhistory_metadata_2[~df_netflixhistory_metadata_2['title'].isin(df_full_user2_profile['title'])]
            df_full_user2_profile = pd.concat([df_full_user2_profile, df_netflix_full_2], ignore_index=True)
            # reset index
            df_full_user2_profile = df_full_user2_profile.reset_index(drop=True)

        if input_type_1 == "netflix":
            df_user1_profile_metadata = df_netflixhistory_metadata_1
            df_user1_profile = df_netflixhistory_metadata_1
            df_full_user1_profile = df_netflixhistory_metadata_1
        if input_type_2 == "netflix":
            df_user2_profile_metadata = df_netflixhistory_metadata_2
            df_user2_profile = df_netflixhistory_metadata_2
            df_full_user2_profile = df_netflixhistory_metadata_2

        #end1 = time.time()
        #print("Scraping time: ", end1-start)

        # Get Taste Match Score
        genre_score = get_genre_match_score(df_user1_profile_metadata, df_user2_profile_metadata)
        movie_score = get_movie_match_score(df_full_user1_profile, df_full_user2_profile)
        fav_genre = get_fav_genre(df_user1_profile_metadata, df_user2_profile_metadata)
        common_movie = get_common_movie(df_user1_profile_metadata, df_user2_profile_metadata)

        expertUserCheck = check_expert_user(df_full_user1_profile, df_full_user2_profile)
        print(expertUserCheck)
        
        with st.spinner('Getting recommendations for both of you'):
            
            # Filter movies based on user input
            df_moviesDB = filter_year_and_rating(df_moviesDB, year_range_filter, rating_filter)
            # Filter movies based on genres
            df_moviesDB = filter_genres(df_moviesDB, genre_filter)

            # Collaborative Filtering - SVD
            # Get User Rating Input DF for SVD
            df_user1RatingSVD = getUserRatingSVDDF(df_movies_db=df_moviesDB, df_user_initial=df_user1_profile, username=99999991)
            df_user2RatingSVD = getUserRatingSVDDF(df_movies_db=df_moviesDB, df_user_initial=df_user2_profile, username=99999992)

            # check if any of df_userRatingSVD is empty (Skip SVD)
            if df_user1RatingSVD.empty or df_user2RatingSVD.empty:
                # add a column SVDRatings as 0 as df_SVD_Results
                df_SVD_Results_1 = df_moviesDB.copy()
                df_SVD_Results_1['SVDRatings'] = 0
                df_SVD_Results_2 = df_moviesDB.copy()
                df_SVD_Results_2['SVDRatings'] = 0
            else:
                # Get Average Rating of User
                averageRating1 = getAverageRating(df_user1RatingSVD)
                averageRating2 = getAverageRating(df_user2RatingSVD)
                # Combine user ratings with rating df
                df_user1RatingSVD = combineRatingsDF(df_user1RatingSVD)
                df_user2RatingSVD = combineRatingsDF(df_user2RatingSVD)

                # Train SVD model
                svd_model_1 = trainSVDModel(df_user1RatingSVD)
                svd_model_2 = trainSVDModel(df_user2RatingSVD)

                # Get List of Sorted SVD Recommendations
                list_SVD_reco_1 = getSVDRecommendations(data=df_user1RatingSVD, df_movies=df_moviesDB, user_id=99999991, algo=svd_model_1)
                list_SVD_reco_2 = getSVDRecommendations(data=df_user2RatingSVD, df_movies=df_moviesDB, user_id=99999992, algo=svd_model_2)

                # Get Dataframe for SVD Recommendations
                df_SVD_Results_1 = get_SVD_Dataframe(sorted_recommendations=list_SVD_reco_1, df_movies=df_moviesDB, user_id=99999991, avgRating=averageRating1)
                df_SVD_Results_2 = get_SVD_Dataframe(sorted_recommendations=list_SVD_reco_2, df_movies=df_moviesDB, user_id=99999992, avgRating=averageRating2)

                # Rearrange svd results based on id
                df_SVD_Results_1 = df_SVD_Results_1.sort_values(by=['id'], ascending = True)
                df_SVD_Results_2 = df_SVD_Results_2.sort_values(by=['id'], ascending = True)

            # Get Group SVD Results
            df_SVD_Results_Group = getGroupSVDResults_2Users(df_SVD_Results_1, df_SVD_Results_2, expertUserCheck)
            
            # Content Based Filtering - CBF
            # Get top genres of users
            top_genres = get_group2_top_genres(df_user1_profile_metadata, df_user2_profile_metadata)
            # Get sorted letterboxd ratings for both users
            sorted_df_letterboxd_1 = sort_rated_df(df_user1_profile_metadata)
            sorted_df_letterboxd_2 = sort_rated_df(df_user2_profile_metadata)
            
            # Get CBF description of movies
            df_SVD_Results_Group = get_CBF_description(df_SVD_Results_Group)
            sorted_df_letterboxd_1 = get_CBF_description_letterboxd(sorted_df_letterboxd_1)
            sorted_df_letterboxd_2 = get_CBF_description_letterboxd(sorted_df_letterboxd_2)

            # Get CBF Input
            df_CBF_input = get_2Group_filtered_CBF_input(df_SVD_Results_Group, sorted_df_letterboxd_1, sorted_df_letterboxd_2, top_genres)

            # Get group sorted group df
            sorted_df_group = get_sorted_2group_letterboxd_df(sorted_df_letterboxd_1, sorted_df_letterboxd_2, expertUserCheck)

            # Get CBF output with cosine similarity
            df_CBF_results = get_group_CBF_cosine_sim(df_movies=df_CBF_input, sorted_df=sorted_df_group)
            # remove duplicates
            df_CBF_results = df_CBF_results.drop_duplicates(subset=['imdb_id'])
            # Get top 20 CBF recommendations
            df_CBF_results = df_CBF_results.head(20)  

            #end2 = time.time()  
            #print("Recommendation time: ", end2-end1)
            #print("Total time: ", end2-start)
        # Reformat columns
        df_CBF_results['languages_list'] = df_CBF_results['spoken_languages'].apply(extract_json_to_list)
        
        # Check username
        if username_1 == "":
            username_1 = "User 1"
        if username_2 == "":
            username_2 = "User 2"

        # Display Group Recommendation Results
        st.header("🗒️ {0} and {1}'s Group Recommendation".format(username_1, username_2))
        # if cbf results is empty
        if df_CBF_results.empty:
            st.error("No recommendations found. You may try again by reducing the restriction in your filtering option.")
            st.stop()
        row_movies = {}
        for i, movie in enumerate(df_CBF_results['title']):
            if (i>=10):
                break
            row = int(i/2)
            if (i%2!=1): 
                row_movies[row] = st.columns(2)
            col = i%2
            # Print details of recommended movie
            url = "https://www.imdb.com/title/" + df_CBF_results['imdb_id'].values[i] +"/"
            row_movies[row][col].subheader("[{0}]({1})".format(movie, url))
            print_year = df_CBF_results['year'].values[i]
            print_year = "{:.0f}".format(print_year)
            row_movies[row][col].write("⭐Average Rating: {0}".format(df_CBF_results['vote_average'].values[i])+ " | 📅Release Year: {0}".format(print_year))
            
            print_genre = ""
            for genre in df_CBF_results['genres_list'].values[i]:
                print_genre += genre + ", "
            print_genre = print_genre[:-2]
            row_movies[row][col].write("🎭Genres: {0}".format(print_genre))
            print_lang = ""
            for lang in df_CBF_results['languages_list'].values[i]:
                print_lang += lang + ", "
            print_lang = print_lang[:-2]
            row_movies[row][col].write("🗣️Languages: {0}".format(print_lang))


            # Display movie poster
            poster_path = get_poster_path(df_CBF_results['imdb_id'].values[i])
            image_url = "https://image.tmdb.org/t/p/original"+poster_path
            

        with st.expander("See full list of recommendations"):
            
            # Drop unnecessary columns in df_CBF_results
            df_CBF_results = df_CBF_results[['title', 'year', 'vote_average', 'genres_list', 'languages_list', 'imdb_id']]
            df_CBF_results = df_CBF_results.reset_index(drop=True)
            st.write("")
            st.dataframe(df_CBF_results)
            def convert_df(df_recom):
                return df_recom.to_csv(index=False).encode('utf-8')
            csv = convert_df(df_CBF_results)
            st.download_button(
                "Download Recommenations",
                csv,
                "{}_Movie Recommendations.csv".format(filename),
                "text/csv",
                key='download-csv'
            )

        st.write("---")
        st.header("📊 Your Group Analysis")
        movie_score = movie_score*1.5
        if movie_score > 50:
            movie_score = 50
        taste_match_score = genre_score + movie_score
        st.write("")
        row_group = st.columns((3,3,3))
        st.subheader("🚀 :blue[Taste Match Score: {0}%]".format(taste_match_score))
        if taste_match_score>80:
            st.subheader("**:green[Perfect Match ]**")
            st.markdown("Both of your tastes in movies aligns flawlessly. You both have an exceptional understanding and appreciation for cinema, making you the 'pitch-perfect' movie-watching duo!")
        elif taste_match_score>65:
            st.subheader("**:green[Hollywood Elite ]**")
            st.markdown("Your taste match scores put you in the league of Hollywood's finest. Like 'Star Wars,' you bring the epicness to your movie choices.")
        else:
            st.subheader("**:green[Silver Screen Explorer ]**")
            st.markdown("You and your movie-watching partner have a good understanding of each other's tastes. You are both willing to explore different genres and discover new movies together, making you much like Katniss and Peeta from 'The Hunger Games.'")
        st.write("")

        if fav_genre != "None":
            st.markdown("Both of you have a penchant for **:green[**{0}** ]** movies and shows!".format(fav_genre))
        st.write("")

        if common_movie != "None":
            st.markdown("A movie or show that brings the both of you together is **:green[**{0}** ]**!".format(common_movie))
        
# Group Recommendation (3 users)
elif reco_type == "Group (3 users)":
    # ----- HEADER -----
    with st.container():
        st.title('🎬 WatchList')
        st.subheader("A Hybrid Recommendation System for Film and Television")
        st.markdown("Select **:green[individual 🧍]** or **:green[group 👪]** recommendation option from the side bar.")
        st.markdown("Provide WatchList with your **:blue[Letterboxd usernames],** or **:blue[ Netflix Viewing History]** (csv file), or **:blue[both]**!")
        with st.expander("More Information"):
            st.write("WatchList is a hybrid recommendation system that combines **:green[collaborative filtering]** and **:green[content-based filtering]** to provide you with movie and TV recommendations.")
            st.write("")
            st.markdown("You may choose to provide WatchList with your **:blue[Letterboxd username]**, **:blue[ Netflix Viewing History]** (csv file), or both.")
            st.write("")
            st.markdown("If you choose to provide your **:blue[Letterboxd username]**, WatchList will scrape your profile and ratings from Letterboxd.")
            st.write("")
            st.markdown("If you choose to provide your **:blue[ Netflix Viewing History]**, WatchList will process your viewing history and scrape the metadata of the movies and TV shows you have watched.")
            st.write("Due to the limitations of Netflix Viewing History, WatchList is only able to process your viewing history"+
                     " if you have watched at least 5 titles using your Netflix account, and may be **:red[inaccurate]** in gathering the correct titles.")
            st.write("")
            st.markdown("**:red[Disclaimer:]** WatchList does not store your Letterboxd username, Netflix Viewing History, or any user data. WatchList is not affiliated with Letterboxd or Netflix.")
            st.markdown("**It will take some time for WatchList to scrape your Letterboxd profile and process your Netflix Viewing History. Please be patient!**")
            st.write("")
        # empty line
        st.write("")
        st.write("")

    # ----- INPUT -----
    with st.container():
        # create two columns for input
        row_input = st.columns((2,1,3,1))
        # username input at column 1
        with row_input[0]:
            # username input
            username_1 = st.text_input('First User\'s Letterboxd Username') 
            st.write("")
            st.write("")
            username_2 = st.text_input('Second User\'s Letterboxd Username') 
            st.write("")
            st.write("")
            username_3 = st.text_input('Third User\'s Letterboxd Username') 
        with row_input[2]:
            # netflixhistory input
            netflixhistory_1 = st.file_uploader('First User\'s Netflix Viewing History', type=['csv'], accept_multiple_files=False, 
                                              help="Download your Netflix Viewing History in csv file format from your account settings.")
            netflixhistory_2 = st.file_uploader('Second User\'s Netflix Viewing History', type=['csv'], accept_multiple_files=False, 
                                              help="Download your Netflix Viewing History in csv file format from your account settings.")
            netflixhistory_3 = st.file_uploader('Third User\'s Netflix Viewing History', type=['csv'], accept_multiple_files=False, 
                                              help="Download your Netflix Viewing History in csv file format from your account settings.")
            
        # place filtering options in a container with smaller width
        st.write("")
        st.write("")
        st.write("")
        with st.container():
            # create two columns for filtering options
            row_filter = st.columns((3,0.7,4,1))
            

            with row_filter[0]:
                # both way slider to filter movies by year
                year_range_filter = st.slider('Filter by year', 1900, 2023, (1900, 2023))
                # both way slider to filter movies by rating
                rating_filter = st.slider('Filter by average rating', min_value=0.0, max_value=10.0, value=(0.0, 10.0), step=0.1)
            with row_filter[2]:
                # filter movies by genre
                genre_filter = st.multiselect('Filter by genre', ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 
                                                                        'Action', 'Crime', 'Thriller', 'Horror', 'History', 
                                                                        'Science Fiction', 'Mystery', 'War', 
                                                                        'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie'], 
                                                                default=['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 
                                                                        'Action', 'Crime', 'Thriller', 'Horror', 'History', 
                                                                        'Science Fiction', 'Mystery', 'War', 
                                                                        'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie'])
        
        st.write("")
        st.write("")
        #netflixhistory = st.file_uploader()
        row_button = st.columns((4,2,2,4)) 
        # submit and reset button in green colour
        submit = row_button[1].button('Submit') 
        reset = row_button[2].button('Reset')
        result = False

    if submit:
        result = True

    if reset:
        # reset all inputs to default
        result = False
    
    if result:
        st.write("---")
        # Get Movie Database as Dataframe
        df_moviesDB = getPyMongoDB("movies_metadata")
        # Clean movie df
        df_moviesDB = clean_movie_df(df_moviesDB)
        today = date.today()
        filename = "{0}_{1}".format(str(today), username_1+"_"+username_2+"_"+username_3)

        # Check input type
        input_type_1 = check_input_type(username_1, netflixhistory_1)
        input_type_2 = check_input_type(username_2, netflixhistory_2)
        input_type_3 = check_input_type(username_3, netflixhistory_3)
        
        # check genre filter
        if len(genre_filter) == 0:
            st.error("Error. Genre filter cannot be left empty.")
            st.stop()
            
        # Check if input_type is Error
        if input_type_1 == "Error" or input_type_2 == "Error" or input_type_3 == "Error":
            st.error("Error. Please input Letterboxd username or Netflix Viewing History.")
            st.stop()
        
        # Process netflix input
        if input_type_1 == "netflix" or input_type_1 == "both":
            # convert netflixhistory to dataframe
            netflixhistory_1 = pd.read_csv(netflixhistory_1)
            # check validity of netflixhistory_1
            validity = check_netflixhistory(netflixhistory_1)
            if validity == "invalid":
                st.error("Error. Netflix Viewing History is invalid. Please try again.")
                st.stop()
            elif validity == "insufficient":
                st.error("Error. Netflix Viewing History is insufficient. Please make sure you have viewed at least 5 titles using your Netflix account.")
                st.stop()
            elif validity == "valid":
                # process netflixhistory
                df_netflixhistory_1 = process_netflixhistory(netflixhistory_1)
                # get netflix metadata
                df_netflixhistory_metadata_1 = getNetflixMetadata(df_netflixhistory_1)
        
        if input_type_2 == "netflix" or input_type_2 == "both":
            # convert netflixhistory to dataframe
            netflixhistory_2 = pd.read_csv(netflixhistory_2)
            # check validity of netflixhistory_2
            validity = check_netflixhistory(netflixhistory_2)
            if validity == "invalid":
                st.error("Error. Netflix Viewing History is invalid. Please try again.")
                st.stop()
            elif validity == "insufficient":
                st.error("Error. Netflix Viewing History is insufficient. Please make sure you have viewed at least 5 titles using your Netflix account.")
                st.stop()
            elif validity == "valid":
                # process netflixhistory
                df_netflixhistory_2 = process_netflixhistory(netflixhistory_2)
                # get netflix metadata
                df_netflixhistory_metadata_2 = getNetflixMetadata(df_netflixhistory_2)
        
        if input_type_3 == "netflix" or input_type_3 == "both":
            # convert netflixhistory to dataframe
            netflixhistory_3 = pd.read_csv(netflixhistory_3)
            # check validity of netflixhistory_3
            validity = check_netflixhistory(netflixhistory_3)
            if validity == "invalid":
                st.error("Error. Netflix Viewing History is invalid. Please try again.")
                st.stop()
            elif validity == "insufficient":
                st.error("Error. Netflix Viewing History is insufficient. Please make sure you have viewed at least 5 titles using your Netflix account.")
                st.stop()
            elif validity == "valid":
                # process netflixhistory
                df_netflixhistory_3 = process_netflixhistory(netflixhistory_3)
                # get netflix metadata
                df_netflixhistory_metadata_3 = getNetflixMetadata(df_netflixhistory_3)
        
        # Process Letterboxd input
        if input_type_1 == "username" or input_type_1 == "both":
            # Start scraping user profile
            with st.spinner('Scraping '+ username_1+'\'s movies on Letterboxd'):
                df_user1_profile = scrape_films_one_page(username_1)
                # check if df_user_profile is empty
                if df_user1_profile.empty:
                    st.error("Error. Username or profile ratings not found. Please try again.")
                    st.stop()
                # get full profile
                df_full_user1_profile = scrape_all_films(username_1)
            with st.spinner('Getting movie details'):
                # Get film metadata through TMDB API    
                df_user1_profile_metadata = getFilmMetadataDF(df_user1_profile)
        if input_type_2 == "username" or input_type_2 == "both":
            with st.spinner('Scraping '+ username_2+'\'s movies on Letterboxd'):
                df_user2_profile = scrape_films_one_page(username_2)
                # check if df_user_profile is empty
                if df_user2_profile.empty:
                    st.error("Error. Username or profile ratings not found. Please try again.")
                    st.stop()
                # get full profile
                df_full_user2_profile = scrape_all_films(username_2)
            with st.spinner('Getting movie details'):
                # Get film metadata through TMDB API    
                df_user2_profile_metadata = getFilmMetadataDF(df_user2_profile)
        if input_type_3 == "username" or input_type_3 == "both":
            with st.spinner('Scraping '+ username_3+'\'s movies on Letterboxd'):
                df_user3_profile = scrape_films_one_page(username_3)
                # check if df_user_profile is empty
                if df_user3_profile.empty:
                    st.error("Error. Username or profile ratings not found. Please try again.")
                    st.stop()
                # get full profile
                df_full_user3_profile = scrape_all_films(username_3)
            
            with st.spinner('Getting movie details'):
                # Get film metadata through TMDB API    
                df_user3_profile_metadata = getFilmMetadataDF(df_user3_profile)
        
        # Check if input_type is both
        if input_type_1 == "both":
            # drop rows in df_netflixhistory_metadata that exists in df_user_profile_metadata
            df_netflixhistory_metadata_1 = df_netflixhistory_metadata_1[~df_netflixhistory_metadata_1['IMDb_ID'].isin(df_user1_profile_metadata['IMDb_ID'])]
            # Combine netflix metadata with letterboxd metadata
            df_user1_profile_metadata = pd.concat([df_user1_profile_metadata, df_netflixhistory_metadata_1], ignore_index=True)

            # drop rows in df_netflixhistory_metadata that exists in df_full_user1_profile
            df_netflix_full_1 = df_netflixhistory_metadata_1[~df_netflixhistory_metadata_1['title'].isin(df_full_user1_profile['title'])]
            df_full_user1_profile = pd.concat([df_full_user1_profile, df_netflix_full_1], ignore_index=True)
            df_full_user1_profile = df_full_user1_profile.reset_index(drop=True)

        if input_type_2 == "both":
            # drop rows in df_netflixhistory_metadata that exists in df_user_profile_metadata
            df_netflixhistory_metadata_2 = df_netflixhistory_metadata_2[~df_netflixhistory_metadata_2['IMDb_ID'].isin(df_user2_profile_metadata['IMDb_ID'])]
            # Combine netflix metadata with letterboxd metadata
            df_user2_profile_metadata = pd.concat([df_user2_profile_metadata, df_netflixhistory_metadata_2], ignore_index=True)

            # drop rows in df_netflixhistory_metadata that exists in df_full_user2_profile
            df_netflix_full_2 = df_netflixhistory_metadata_2[~df_netflixhistory_metadata_2['title'].isin(df_full_user2_profile['title'])]
            df_full_user2_profile = pd.concat([df_full_user2_profile, df_netflix_full_2], ignore_index=True)
            df_full_user2_profile = df_full_user2_profile.reset_index(drop=True)

        if input_type_3 == "both":
            # drop rows in df_netflixhistory_metadata that exists in df_user_profile_metadata
            df_netflixhistory_metadata_3 = df_netflixhistory_metadata_3[~df_netflixhistory_metadata_3['IMDb_ID'].isin(df_user3_profile_metadata['IMDb_ID'])]
            # Combine netflix metadata with letterboxd metadata
            df_user3_profile_metadata = pd.concat([df_user3_profile_metadata, df_netflixhistory_metadata_3], ignore_index=True)

            # drop rows in df_netflixhistory_metadata that exists in df_full_user3_profile
            df_netflix_full_3 = df_netflixhistory_metadata_3[~df_netflixhistory_metadata_3['title'].isin(df_full_user3_profile['title'])]
            df_full_user3_profile = pd.concat([df_full_user3_profile, df_netflix_full_3], ignore_index=True)
            df_full_user3_profile = df_full_user3_profile.reset_index(drop=True)
        
        if input_type_1 == "netflix":
            df_user1_profile_metadata = df_netflixhistory_metadata_1
            df_user1_profile = df_netflixhistory_metadata_1
            df_full_user1_profile = df_netflixhistory_metadata_1
        if input_type_2 == "netflix":
            df_user2_profile_metadata = df_netflixhistory_metadata_2
            df_user2_profile = df_netflixhistory_metadata_2
            df_full_user2_profile = df_netflixhistory_metadata_2
        if input_type_3 == "netflix":
            df_user3_profile_metadata = df_netflixhistory_metadata_3
            df_user3_profile = df_netflixhistory_metadata_3
            df_full_user3_profile = df_netflixhistory_metadata_3
        
        # Get Taste Match Score
        genre_score = get_genre_match_score_3users(df_user1_profile_metadata, df_user2_profile_metadata, df_user3_profile_metadata)
        movie_score = get_movie_match_score_3users(df_full_user1_profile, df_full_user2_profile, df_full_user3_profile)
        fav_genre = get_fav_genre_3users(df_user1_profile_metadata, df_user2_profile_metadata, df_user3_profile_metadata)
        common_movie = get_common_movie_3users(df_user1_profile_metadata, df_user2_profile_metadata, df_user3_profile_metadata)

        expertUserCheck = check_expert_user_3users(df_user1_profile_metadata, df_user2_profile_metadata, df_user3_profile_metadata)
        print(expertUserCheck)
        with st.spinner('Getting recommendations for your group'):
            
            # Filter movies based on user input
            df_moviesDB = filter_year_and_rating(df_moviesDB, year_range_filter, rating_filter)
            # Filter movies based on genres
            df_moviesDB = filter_genres(df_moviesDB, genre_filter)

            # Collaborative Filtering - SVD
            # Get User Rating Input DF for SVD
            df_user1RatingSVD = getUserRatingSVDDF(df_movies_db=df_moviesDB, df_user_initial=df_user1_profile, username=99999991)
            df_user2RatingSVD = getUserRatingSVDDF(df_movies_db=df_moviesDB, df_user_initial=df_user2_profile, username=99999992)
            df_user3RatingSVD = getUserRatingSVDDF(df_movies_db=df_moviesDB, df_user_initial=df_user3_profile, username=99999993)

            # check if any of df_userRatingSVD is empty (Skip SVD)
            if df_user1RatingSVD.empty or df_user2RatingSVD.empty or df_user3RatingSVD.empty:
                # add a column SVDRatings as 0 as df_SVD_Results
                df_SVD_Results_1 = df_moviesDB.copy()
                df_SVD_Results_1['SVDRatings'] = 0
                df_SVD_Results_2 = df_moviesDB.copy()
                df_SVD_Results_2['SVDRatings'] = 0
                df_SVD_Results_3 = df_moviesDB.copy()
                df_SVD_Results_3['SVDRatings'] = 0
            else:
                # get average rating of user
                averageRating1 = getAverageRating(df_user1RatingSVD)
                averageRating2 = getAverageRating(df_user2RatingSVD)
                averageRating3 = getAverageRating(df_user3RatingSVD)

                # Combine user ratings with rating df
                df_user1RatingSVD = combineRatingsDF(df_user1RatingSVD)
                df_user2RatingSVD = combineRatingsDF(df_user2RatingSVD)
                df_user3RatingSVD = combineRatingsDF(df_user3RatingSVD)

                # Train SVD model
                svd_model_1 = trainSVDModel(df_user1RatingSVD)
                svd_model_2 = trainSVDModel(df_user2RatingSVD)
                svd_model_3 = trainSVDModel(df_user3RatingSVD)

                # Get List of Sorted SVD Recommendations
                list_SVD_reco_1 = getSVDRecommendations(data=df_user1RatingSVD, df_movies=df_moviesDB, user_id=99999991, algo=svd_model_1)
                list_SVD_reco_2 = getSVDRecommendations(data=df_user2RatingSVD, df_movies=df_moviesDB, user_id=99999992, algo=svd_model_2)
                list_SVD_reco_3 = getSVDRecommendations(data=df_user3RatingSVD, df_movies=df_moviesDB, user_id=99999993, algo=svd_model_3)

                # Get Dataframe for SVD Recommendations
                df_SVD_Results_1 = get_SVD_Dataframe(sorted_recommendations=list_SVD_reco_1, df_movies=df_moviesDB, user_id=99999991, avgRating=averageRating1)
                df_SVD_Results_2 = get_SVD_Dataframe(sorted_recommendations=list_SVD_reco_2, df_movies=df_moviesDB, user_id=99999992, avgRating=averageRating2)
                df_SVD_Results_3 = get_SVD_Dataframe(sorted_recommendations=list_SVD_reco_3, df_movies=df_moviesDB, user_id=99999993, avgRating=averageRating3)

                # Rearrange svd results based on id
                df_SVD_Results_1 = df_SVD_Results_1.sort_values(by=['id'], ascending = True)
                df_SVD_Results_2 = df_SVD_Results_2.sort_values(by=['id'], ascending = True)
                df_SVD_Results_3 = df_SVD_Results_3.sort_values(by=['id'], ascending = True)

            # Get Group SVD Results
            df_SVD_Results_Group = getGroupSVDResults_3Users(df_SVD_Results_1, df_SVD_Results_2, df_SVD_Results_3, expertUserCheck)
            
            # Content Based Filtering - CBF
            # Get top genres of users
            top_genres = get_group3_top_genres(df_user1_profile_metadata, df_user2_profile_metadata, df_user3_profile_metadata)
            # Get sorted letterboxd ratings for both users
            sorted_df_letterboxd_1 = sort_rated_df(df_user1_profile_metadata)
            sorted_df_letterboxd_2 = sort_rated_df(df_user2_profile_metadata)
            sorted_df_letterboxd_3 = sort_rated_df(df_user3_profile_metadata)

            # Get CBF description of movies
            df_SVD_Results_Group = get_CBF_description(df_SVD_Results_Group)
            sorted_df_letterboxd_1 = get_CBF_description_letterboxd(sorted_df_letterboxd_1)
            sorted_df_letterboxd_2 = get_CBF_description_letterboxd(sorted_df_letterboxd_2)
            sorted_df_letterboxd_3 = get_CBF_description_letterboxd(sorted_df_letterboxd_3)

            # Get CBF Input
            df_CBF_input = get_3Group_filtered_CBF_input(df_SVD_Results_Group, sorted_df_letterboxd_1, sorted_df_letterboxd_2, sorted_df_letterboxd_3, top_genres)

            # Get group sorted group df
            sorted_df_group = get_sorted_3group_letterboxd_df(sorted_df_letterboxd_1, sorted_df_letterboxd_2, sorted_df_letterboxd_3, expertUserCheck)

            # Get CBF output with cosine similarity
            df_CBF_results = get_group_CBF_cosine_sim(df_movies=df_CBF_input, sorted_df=sorted_df_group)
            # remove duplicates
            df_CBF_results = df_CBF_results.drop_duplicates(subset=['imdb_id'])
            # Get top 20 CBF recommendations
            df_CBF_results = df_CBF_results.head(20)  

            
        # Reformat columns
        df_CBF_results['languages_list'] = df_CBF_results['spoken_languages'].apply(extract_json_to_list)
        

        # Check username
        if username_1 == "":
            username_1 = "User 1"
        if username_2 == "":
            username_2 = "User 2"
        if username_3 == "":
            username_3 = "User 3"
        # Display Group Recommendation Results
        st.header("🗒️ {0}, {1}, and {2}'s Group Recommendation".format(username_1, username_2, username_3))
        # if cbf results is empty
        if df_CBF_results.empty:
            st.error("No recommendations found. You may try again by reducing the restriction in your filtering option.")
            st.stop()
        row_movies = {}
        for i, movie in enumerate(df_CBF_results['title']):
            if (i>=10):
                break
            row = int(i/2)
            if (i%2!=1): 
                row_movies[row] = st.columns(2)
            col = i%2
            # Print details of recommended movie
            url = "https://www.imdb.com/title/" + df_CBF_results['imdb_id'].values[i] +"/"
            row_movies[row][col].subheader("[{0}]({1})".format(movie, url))
            print_year = df_CBF_results['year'].values[i]
            print_year = "{:.0f}".format(print_year)
            row_movies[row][col].write("⭐Average Rating: {0}".format(df_CBF_results['vote_average'].values[i])+" | 📅Release Year: {0}".format(print_year))
            print_genre = ""
            for genre in df_CBF_results['genres_list'].values[i]:
                print_genre += genre + ", "
            print_genre = print_genre[:-2]
            row_movies[row][col].write("🎭Genres: {0}".format(print_genre))
            print_lang = ""
            for lang in df_CBF_results['languages_list'].values[i]:
                print_lang += lang + ", "
            print_lang = print_lang[:-2]
            row_movies[row][col].write("🗣️Languages: {0}".format(print_lang))

            # Display movie poster
            poster_path = get_poster_path(df_CBF_results['imdb_id'].values[i])
            image_url = "https://image.tmdb.org/t/p/original"+poster_path
            

        with st.expander("See full list of recommendations"):
            
            # get necessary columns for display
            df_CBF_results = df_CBF_results[['title', 'year', 'vote_average', 'genres_list', 'languages_list', 'imdb_id']]
            # reset index
            df_CBF_results = df_CBF_results.reset_index(drop=True)
            st.write("")
            st.dataframe(df_CBF_results)
            def convert_df(df_recom):
                return df_recom.to_csv(index=False).encode('utf-8')
            csv = convert_df(df_CBF_results)
            st.download_button(
                "Download Recommenations",
                csv,
                "{}_Movie Recommendations.csv".format(filename),
                "text/csv",
                key='download-csv'
            )
        
        st.write("---")
        st.header("📊 Your Group Analysis")
        movie_score = movie_score*1.5
        if movie_score > 50:
            movie_score = 50
        taste_match_score = genre_score + movie_score
        st.write("")
        row_group = st.columns((3,3,3))
        st.subheader("🚀 :blue[Taste Match Score: {0}%]".format(taste_match_score))
        if taste_match_score>75:
            st.subheader("**:green[Perfect Match ]**")
            st.markdown("All three of your tastes in movies aligns flawlessly. You have an exceptional understanding and appreciation for cinema, making you the 'pitch-perfect' movie-watching trio!")
        elif taste_match_score>55:
            st.subheader("**:green[Hollywood Elite ]**")
            st.markdown("Your taste match scores put your group in the league of Hollywood's finest. Like 'Star Wars,' you bring the epicness to your content choices.")
        else:
            st.subheader("**:green[Silver Screen Explorer ]**")
            st.markdown("You and your movie-watching partners have a good understanding of each other's tastes. You are all willing to explore different genres and discover new movies together, making you much like Harry, Ron, and Hermione from 'Harry Potter'.")
        st.write("")
        if fav_genre != "None":
            st.markdown("All three of you have a penchant for **:green[**{0}** ]** movies and shows!".format(fav_genre))
        st.write("")

        if common_movie != "None":
            st.markdown("A movie or show that brings the three of you together is **:green[**{0}** ]**!".format(common_movie))
        