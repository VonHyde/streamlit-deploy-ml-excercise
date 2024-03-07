import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el modelo, el vectorizador y la data

with open("src/knn_neighbors-6_algorithm-brute_metric-cosine.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("src/total_data.sav", "rb") as f:
    total_data = pickle.load(f)
    
vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', lowercase=True)
matrix = vectorizer.fit_transform(total_data['tags'])


#Definir la aplicacion del modelo

def get_movie_recommendations(movie_title):
    movie_index = total_data[total_data["title"] == movie_title].index[0]
    distances, indices = knn_model.kneighbors(matrix[movie_index])
    similar_movies = [(total_data["title"][i], distances[0][j]) for j, i in enumerate(indices[0])]

    return similar_movies[1:]

# Definir la aplicación Streamlit
def main():
    st.title('Movie Recommendation System')

    # Obtener la lista de títulos de películas del conjunto de datos
    movie_titles = total_data["title"].tolist()

    # Interfaz de usuario para seleccionar un título de película de la lista desplegable
    movie_title = st.selectbox('Select a movie title', movie_titles)

    # Obtener recomendaciones cuando se selecciona un título de película
    if st.button('Get Recommendations'):
        recommendations = get_movie_recommendations(movie_title)
        st.write("Recommended movies for '{}':".format(movie_title))
        for movie, distances in recommendations:
            st.write("- Movie: {}".format(movie))

if __name__ == '__main__':
    main()