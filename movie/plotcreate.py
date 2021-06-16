import pandas as pd
import numpy as np
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

movie_overviews = pd.read_csv('movie_overviews.csv')
metadata=pd.DataFrame(movie_overviews.drop(['overview','id'],axis=1))
movie_plots= pd.Series(movie_overviews.overview)


#print(metadata)

# Generate mapping between titles and index
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# Initialize the TfidfVectorizer 
tfidf = TfidfVectorizer(stop_words='english')

# Construct the TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_plots.apply(lambda x: np.str_(x)))

# Generate the cosine similarity matrix
sim = cosine_similarity(tfidf_matrix)

metadata.to_csv('metadata.csv',index=False)