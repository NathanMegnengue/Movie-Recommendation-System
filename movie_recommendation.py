# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger les données
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.item'
movies = pd.read_csv(url, sep='|', encoding='latin-1', header=None)
column_names = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies.columns = column_names

# Afficher les premières lignes du dataset
print(movies.head())

# Créer une nouvelle colonne 'combined_features' en combinant les colonnes de genres
genre_columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies['combined_features'] = movies[genre_columns].apply(lambda x: ' '.join(x.index[x == 1]), axis=1)

# Créer une matrice de comptage
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['combined_features'])

# Calculer la similarité cosinus
cosine_sim = cosine_similarity(count_matrix)

# Fonction pour obtenir l'indice d'un film à partir de son titre
def get_index_from_title(title):
    return movies[movies['movie_title'] == title].index.values[0]

# Fonction pour obtenir le titre d'un film à partir de son indice
def get_title_from_index(index):
    return movies.iloc[index]['movie_title']

# Fonction de recommandation
def get_recommendations(movie_title):
    try:
        movie_index = get_index_from_title(movie_title)
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]
        
        print(f"Les films similaires à {movie_title} sont :\n")
        for i in sorted_similar_movies:
            print(get_title_from_index(i[0]))
    except IndexError:
        print(f"Le film '{movie_title}' n'a pas été trouvé dans la base de données.")

# Exemple : recommander des films similaires à "Toy Story (1995)"
get_recommendations('Aladdin (1992)')

