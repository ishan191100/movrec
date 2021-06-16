import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

# define a function that creates similarity matrix
# if it doesn't exist
def create_sim():
    metadata = pd.read_csv('metadata.csv')
    movie_overviews=pd.read_csv('movie_overviews.csv')
    movie_plots= pd.Series(movie_overviews.overview)

    # Initialize the TfidfVectorizer 
    tfidf = TfidfVectorizer(stop_words='english')

    # Construct the TF-IDF matrix
    tfidf_matrix = tfidf.fit_transform(movie_overviews['overview'].apply(lambda x: np.str_(x)))

    # Generate the cosine similarity matrix
    sim = cosine_similarity(tfidf_matrix)

    # Generate recommendations 
    return movie_overviews,sim


# defining a function that recommends 10 most similar movies
def rcmd(movie):
    # check if data and sim are already assigned
    try:
        movie_overviews.head()
        sim.shape
    except:
        movie_overviews, sim = create_sim()
    # check if the movie is in our database or not
    if movie not in movie_overviews['title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
        # getting the index of the movie in the dataframe
        i = movie_overviews.loc[movie_overviews['title']==movie].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(movie_overviews['title'][a])
        return l

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')



if __name__ == '__main__':
    app.run()
