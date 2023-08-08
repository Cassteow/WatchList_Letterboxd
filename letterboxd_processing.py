from bs4 import BeautifulSoup
import streamlit as st
import requests
import pandas as pd
import concurrent.futures
import json
import re


LETTERBOXD_DOMAIN = "https://letterboxd.com"
headers = {
     "accept": "application/json",
     "Authorization": "Bearer " + st.secrets['tmdb_key']
}



# Function to Transform Stars into Numerical Ratings
def transform_ratings(original_rating):
    """
    transforms raw star rating into float value
    :param: some_str: actual star rating
    :rtype: returns the float representation of the given star(s)
    """
    stars = {
        "★": 1,
        "★★": 2,
        "★★★": 3,
        "★★★★": 4,
        "★★★★★": 5,
        "½": 0.5,
        "★½": 1.5,
        "★★½": 2.5,
        "★★★½": 3.5,
        "★★★★½": 4.5
    }
    try:
        return stars[original_rating]
    except:
        return -1

# Function to Get IMDb ID of Movies or TV Shows with Letterboxd URL
def get_imdb_id(letterboxd_url):
    respond = {}
    resp = requests.get(letterboxd_url)
    if resp.status_code != 200:
        respond = ("None", letterboxd_url)
        return respond
    # Extract the IMDb URL
    re_match = re.findall(r'href=".+title/(tt\d+)/maindetails"', resp.text)
    if not re_match:
        respond = ("None", letterboxd_url)
        return respond
    # return IMDB ID and letterboxd_url as a tuple
    respond = (re_match[0], letterboxd_url)
    return respond

# Function to send concurrent requests to get IMDB ID with df_film as input
def get_imdb_ids_concurrently(df):
    imdb_ids = []
    urls = df['letterboxd_link'].tolist()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks for each URL
        futures = [executor.submit(get_imdb_id, url) for url in urls]

        # Retrieve results as they complete
        for future in concurrent.futures.as_completed(futures):
            imdb_id = future.result()
            imdb_ids.append(imdb_id)

    return imdb_ids


# Function to Scrape All Watched and Rated Films/TV of user (without IMDb ID)
def scrape_all_films(username):
    movies_dict = {}
    movies_dict['letterboxd_id'] = []
    movies_dict['title'] = []
    movies_dict['rating'] = []
    movies_dict['liked'] = []
    movies_dict['letterboxd_link'] = []

    url = LETTERBOXD_DOMAIN + "/" + username + "/films/by/entry-rating/"
    url_page = requests.get(url)
    soup = BeautifulSoup(url_page.content, 'lxml')
    
    # Check if the page is valid
    if url_page.status_code != 200:
        return pd.DataFrame(movies_dict)
    
    # check number of pages
    li_pagination = soup.findAll("li", {"class": "paginate-page"})
    if len(li_pagination) == 0:
        ul = soup.find("ul", {"class": "poster-list"})
        if (ul != None):
            movies = ul.find_all("li")
            for movie in movies:
                print("Scraping: "+movie.find('img')['alt'])
                letterboxd_link = movie.find('div')['data-target-link']
                movies_dict['letterboxd_id'].append(movie.find('div')['data-film-id'])
                movies_dict['title'].append(movie.find('img')['alt'])
                movies_dict['rating'].append(transform_ratings(movie.find('p', {"class": "poster-viewingdata"}).get_text().strip()))
                movies_dict['liked'].append(movie.find('span', {'class': 'like'})!=None)
                movies_dict['letterboxd_link'].append(LETTERBOXD_DOMAIN+letterboxd_link)
                
    else:
        for i in range(len(li_pagination)):
            url = LETTERBOXD_DOMAIN + "/" + username + "/films/by/entry-rating/page/" + str(i+1)
            url_page = requests.get(url)
            
            if url_page.status_code != 200:
                return pd.DataFrame(movies_dict)
            
            soup = BeautifulSoup(url_page.content, 'lxml')
            ul = soup.find("ul", {"class": "poster-list"})
            if (ul != None):
                movies = ul.find_all("li")
                for movie in movies:
                    print("Scraping: "+movie.find('img')['alt'])
                    letterboxd_link = movie.find('div')['data-target-link']
                    movies_dict['letterboxd_id'].append(movie.find('div')['data-film-id'])
                    movies_dict['title'].append(movie.find('img')['alt'])
                    movies_dict['rating'].append(transform_ratings(movie.find('p', {"class": "poster-viewingdata"}).get_text().strip()))
                    movies_dict['liked'].append(movie.find('span', {'class': 'like'})!=None)
                    movies_dict['letterboxd_link'].append(LETTERBOXD_DOMAIN+letterboxd_link)
                    
    
    df_film = pd.DataFrame(movies_dict)  
    # Drop Films with No Rating
    #df_film = df_film[df_film['rating']!=-1].reset_index(drop=True)  
    return df_film

# Function scrape one page
def scrape_films_one_page(username):
    movies_dict = {}
    movies_dict['letterboxd_id'] = []
    movies_dict['title'] = []
    movies_dict['rating'] = []
    movies_dict['liked'] = []
    movies_dict['letterboxd_link'] = []

    url = LETTERBOXD_DOMAIN + "/" + username + "/films/by/entry-rating/"
    url_page = requests.get(url)

    # Check if the page is valid
    if url_page.status_code != 200:
        return pd.DataFrame(movies_dict)
    soup = BeautifulSoup(url_page.content, 'lxml')
    
    ul = soup.find("ul", {"class": "poster-list"})
    if (ul != None):
        movies = ul.find_all("li")
        progress = 0
        bar = st.progress(progress)
        for movie in movies:
            progress = progress+1
            print("Scraping movie (one page): "+movie.find('img')['alt'])
            # Get movie details
            letterboxd_link = movie.find('div')['data-target-link']
            movies_dict['letterboxd_id'].append(movie.find('div')['data-film-id'])
            movies_dict['title'].append(movie.find('img')['alt'])
            movies_dict['rating'].append(transform_ratings(movie.find('p', {"class": "poster-viewingdata"}).get_text().strip()))
            movies_dict['liked'].append(movie.find('span', {'class': 'like'})!=None)
            movies_dict['letterboxd_link'].append(LETTERBOXD_DOMAIN+letterboxd_link)
            bar.progress(progress/len(movies))
        bar.empty()
    df_film = pd.DataFrame(movies_dict)  
    # Drop Films with No Rating
    df_film = df_film[df_film['rating']!=-1].reset_index(drop=True) 
    # Get IMDB IDs
    imdb_list = get_imdb_ids_concurrently(df_film) 
    df_film['IMDb_ID'] = ""
    for index, row in df_film.iterrows():
        for imdb in imdb_list:
            if row['letterboxd_link'] == imdb[1]:
                df_film.at[index, 'IMDb_ID'] = imdb[0]

    # remove rows with no IMDb_ID
    df_film = df_film[df_film['IMDb_ID']!="None"].reset_index(drop=True)
    return df_film

# Function to get Film Details using TMDb API
def getFilmDetailsTMDb(imdbId):
    # Get URL Prompt to access TMDb API
    film_details_TMDb_url = "https://api.themoviedb.org/3/find/" + imdbId + "?external_source=imdb_id"
    
    # Get response from TMDb API
    response = requests.get(film_details_TMDb_url, headers=headers)
    return response.text

# Function to Get Movie Genre List
def getMovieGenreList():
    movie_genre_list_url = "https://api.themoviedb.org/3/genre/movie/list"
    response = requests.get(movie_genre_list_url, headers=headers)
    return response.text

# Function to Get TV Genre List
def getTVGenreList():
    tv_genre_list_url = "https://api.themoviedb.org/3/genre/tv/list" 
    response = requests.get(tv_genre_list_url, headers=headers)
    return response.text

# Function to Get Genre Name Using ID
def getGenresFromID(jsonGenreList, genre_ids_list):
    # Create List for Genre Names
    genres = ""
    genreList = json.loads(jsonGenreList)
    genreList = genreList["genres"]
    
    #Find Genre names by IDs
    for id in genre_ids_list:
        for genre in genreList:
            if genre["id"] == id:
                genres = genres + genre["name"] +"," 
    return genres[:-1] # remove last comma


# Function to Create DF with Film Metadata
def getFilmMetadataDF(film_df):
    adult = []
    original_language = []
    overview = []
    vote_average = []
    vote_count = []
    popularity = []
    release_date = [] # For TV (First Release Date)
    poster_path = []
    genres = []
    media = []
    tmdb_id = []
    # Get Genre List
    tVGenreList = getTVGenreList()
    movieGenreList = getMovieGenreList()

    progress = 0
    bar = st.progress(progress)
    for index, row in film_df.iterrows():
        progress = progress+1
        with st.spinner('Getting movie details: '+row['title']):
            # Get IMDb ID for film
            imdbID = row['IMDb_ID']
            print("Getting details for: "+row['title'] + " (" + imdbID + ")" )
        
            # Call function to get Film Details
            tmdbResponse = getFilmDetailsTMDb(imdbID)
            filmDetail = json.loads(tmdbResponse)
            # For Movies
            if(len(filmDetail['movie_results'])!=0):
                adult.append(str(filmDetail['movie_results'][0]['adult']))
                original_language.append(str(filmDetail['movie_results'][0]['original_language']))
                overview.append(str(filmDetail['movie_results'][0]['overview'].replace('\n', ' ')))
                vote_average.append(str(filmDetail['movie_results'][0]['vote_average']))
                vote_count.append(str(filmDetail['movie_results'][0]['vote_count']))
                popularity.append(filmDetail['movie_results'][0]['popularity'])
                release_date.append(filmDetail['movie_results'][0]['release_date'])
                
                movieGenreIDs = filmDetail['movie_results'][0]['genre_ids']
                movieGenres = getGenresFromID(movieGenreList, movieGenreIDs)
                genres.append(movieGenres)
                poster_path.append(str(filmDetail['movie_results'][0]['poster_path']))
                media.append("movie")
                tmdb_id.append(filmDetail['movie_results'][0]['id'])
            
            # For TV
            elif(len(filmDetail['tv_results'])!=0):
                adult.append(str(filmDetail['tv_results'][0]['adult']))
                original_language.append(str(filmDetail['tv_results'][0]['original_language']))
                overview.append(str(filmDetail['tv_results'][0]['overview'].replace('\n', ' ')))
                vote_average.append(str(filmDetail['tv_results'][0]['vote_average']))
                vote_count.append(str(filmDetail['tv_results'][0]['vote_count']))
                popularity.append(filmDetail['tv_results'][0]['popularity'])
                release_date.append(filmDetail['tv_results'][0]['first_air_date'])
                
                tvGenreIDs = filmDetail['tv_results'][0]['genre_ids']
                tvGenres = getGenresFromID(tVGenreList, tvGenreIDs)
                genres.append(tvGenres)
                poster_path.append(str(filmDetail['tv_results'][0]['poster_path']))
                media.append("TV")
                tmdb_id.append(filmDetail['tv_results'][0]['id'])

            else:
                errorString = ""
                adult.append(errorString)
                original_language.append(errorString)
                overview.append(errorString)
                vote_average.append(errorString)
                vote_count.append(errorString)
                popularity.append(errorString)
                release_date.append(errorString)
                genres.append(errorString)
                poster_path.append(errorString)
                media.append(errorString)
                tmdb_id.append(errorString)
        bar.progress(progress/len(film_df))
    bar.empty()
    # Add the new_ratings list as a new column to the dataframe
    film_df['adult'] = adult
    film_df['original_language'] = original_language
    film_df['overview'] = overview
    film_df['vote_average'] = vote_average
    film_df['vote_count'] = vote_count
    film_df['popularity'] = popularity
    film_df['release_date'] = release_date
    film_df['genres'] = genres
    film_df["poster_path"] = poster_path
    film_df["media"] = media
    film_df["tmdb_id"] = tmdb_id
    # drop rows with no tmdb_id
    film_df = film_df[film_df['tmdb_id']!=""].reset_index(drop=True)
    return film_df

# Function to get poster path of movie with imdb_id
def get_poster_path(imdb_id):
    # Get URL Prompt to access TMDb API
    film_details_TMDb_url = "https://api.themoviedb.org/3/find/" + imdb_id + "?external_source=imdb_id"
    
    # Get response from TMDb API
    response = requests.get(film_details_TMDb_url, headers=headers)
    filmDetail = json.loads(response.text)
    if(len(filmDetail['movie_results'])!=0):
        return filmDetail['movie_results'][0]['poster_path']
    elif(len(filmDetail['tv_results'])!=0):
        return filmDetail['tv_results'][0]['poster_path']
    else:
        return ""

