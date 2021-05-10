# ------------------------------------------ Import ------------------------------------------

# Importing the libraries

from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
from functions import text_cleaning, recommend, model_used
from preprocessing import listings, list_stopwords


# ------------------------------------------ Code ------------------------------------------

st.title("Moteur de recommandation d'aliments")
model = st.sidebar.radio('Modèle', ['Tfidf', 'CountVectorizer', 'BERT'])
entree = st.sidebar.text_area("Entrez votre recherche :", height=50)
num = st.sidebar.slider('Nombre de résultats affichés :', min_value=1, max_value=100)
with st.sidebar.beta_expander('Affichage'):
    ing = st.checkbox('Ingrédients')
    ale = st.checkbox('Allergènes')
    mar = st.checkbox('Marque')
    nut = st.checkbox('Nutriscore')
    pal = st.checkbox("Présence d'huile de palme")
    img = st.checkbox('Image')
    valnut = st.checkbox('Valeurs nutritionnelles')
    sim = st.checkbox('Score de similarité')
filtnut = st.sidebar.multiselect('Nutriscore', listings['nutrition_grade_fr'].unique())
params = (ing, ale, mar, nut, pal, img, valnut, sim)
mod, matrix = model_used(df=listings, model=model, stop=list_stopwords)

if entree:
    entree_cleared = text_cleaning(entree)

    if model == 'BERT':
        search = mod.encode([entree_cleared])
    else:
        search = mod.transform([entree_cleared])

    cosine_similarities = linear_kernel(search, matrix, dense_output=True).astype('float16')

    results = {}
    similar_indices = cosine_similarities[0].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[0][i], i) for i in similar_indices]
    results[entree] = similar_items

    recommend(itemname=entree, num=num, results=results, df=listings, param=params, filtre=filtnut)
