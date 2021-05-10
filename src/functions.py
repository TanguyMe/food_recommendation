# ------------------------------------------ Import ------------------------------------------

# Importing the libraries
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer

# ------------------------------------------ Cleaning ------------------------------------------


def remove_ponctuation(text):
    """Remove punctuation from text"""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_space(text):
    """Remove double and more space from text"""
    text = text.strip()
    text = text.split()
    return " ".join(text)


def remove_numbers(text):
    """Remove independant numbers from text"""
    text = [word for word in text.split() if not word.isdigit()]
    return " ".join(text)


def remove_any_numbers(text):
    """Remove all independant numbers and all numbers contained in other words"""
    text = [word for word in text.split() if not any(c.isdigit() for c in word)]
    return " ".join(text)


def tokenize(text):
    """Tokenize text"""
    tokens = word_tokenize(text)
    return tokens


def stemming(text):
    """Apply french stemmer to text (remove suffix)"""
    stemmer = FrenchStemmer()
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)


def lemmatizer(text):
    """Apply Wordnet lemmatizer to text (go to root word)"""
    wnl = WordNetLemmatizer()
    text = [wnl.lemmatize(word) for word in text.split()]
    return " ".join(text)


def text_cleaning(text):
    """Apply all text cleaning needed to text"""
    text = remove_ponctuation(text)
    text = remove_space(text)
    text = remove_any_numbers(text)
    return text


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def model_used(df, model, stop):
    """Given a model choice, return the model and the computed matrix"""
    if model == 'Tfidf':
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words=stop)
        tfidf_matrix = tf.fit_transform(df['content'])
        return tf, tfidf_matrix
    elif model == 'CountVectorizer':
        cv = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words=stop)
        matrix = cv.fit_transform(df['content'])
        return cv, matrix
    elif model == 'BERT':
        bert = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        matrix = bert.encode(df['content'].astype('str'), show_progress_bar=True)
        return bert, matrix


# ------------------------------------------ User ------------------------------------------


def item(id, df):
    """Return product name from id"""
    name = df['product_name'][id].split(' // ')[0]
    return name


def recommend(itemname, num, results, df, param, filtre):
    """Display the recommandations for a given item with the wanted parameters.
    User can also apply a filter on nutriscore via the web app"""
    st.header('Recommandation de ' + str(num) + ' produits similaires à ' + itemname)
    st.header('     -------------------------------------------------------------       ')
    recs = []
    indice = 0
    # Take the wanted number of recommandations (and apply condition if wanted)
    while len(recs) <= num:
        if len(filtre) != 0:
            if df['nutrition_grade_fr'][results[itemname][indice][1]] in filtre:
                recs.append(results[itemname][indice])
        else:
            recs.append(results[itemname][indice])
        indice = indice+1
    # Display all the recommandations
    for rec in recs:
        st.subheader(item(rec[1], df))
        if any(param):
            with st.beta_expander('Détails', expanded=True):
                # params = (ing, ale, mar, nut, pal, img, valnut, sim)
                # Marque
                if param[2]:
                    try:
                        st.write('\nMarque: ' + df['brands'][rec[1]])
                    except:
                        pass
                # Image
                if param[5]:
                    try:
                        st.image(df['image_url'][rec[1]], width=200)
                    except:
                        pass
                # Nutriscore
                if param[3]:
                    try:
                        if df['nutrition_grade_fr'][rec[1]] == 'a':
                            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Nutri-score-A.svg/120px-Nutri-score-A.svg.png')
                        elif df['nutrition_grade_fr'][rec[1]] == 'b':
                            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Nutri-score-B.svg/120px-Nutri-score-B.svg.png')
                        elif df['nutrition_grade_fr'][rec[1]] == 'c':
                            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Nutri-score-C.svg/120px-Nutri-score-C.svg.png')
                        elif df['nutrition_grade_fr'][rec[1]] == 'd':
                            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/d/d6/Nutri-score-D.svg/120px-Nutri-score-D.svg.png')
                        elif df['nutrition_grade_fr'][rec[1]] == 'e':
                            st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Nutri-score-E.svg/120px-Nutri-score-E.svg.png')
                    except:
                        st.write('\nLe nutriscore de ' + item(rec[1], df) + " n'est pas renseigné")
                # Ingrédients
                if param[0]:
                    try:
                        st.write('\nIngrédients: ' + df['ingredients_text'][rec[1]])
                    except:
                        st.write('\nIngrédients non renseignés')
                # Allergènes
                if param[1]:
                    try:
                        if not(isinstance(df['allergens'][rec[1]], str) or isinstance(df['traces'][rec[1]], str)):
                            st.write("\nPas d'allergènes renseignés")
                        if isinstance(df['allergens'][rec[1]], str):
                            st.write('\nAllergènes: ' + str(df['allergens'][rec[1]]))
                        if isinstance(df['traces'][rec[1]], str):
                            st.write("\nTraces d'allergènes: " + str(df['traces'][rec[1]]))
                    except:
                        st.write("\nPas d'allergènes renseignés")
                # Huile de palme
                if param[4]:
                    try:
                        if df['ingredients_from_palm_oil_n'][rec[1]] != 0:
                            st.write("Présence d'huile de palme")
                        else:
                            st.write("Pas d'huile de palme")
                    except:
                        st.write("Pas d'huile de palme")
                # Valeurs nutritionnelles
                if param[6]:
                    nutcol=['energy_100g',
                            'saturated-fat_100g', 'sugars_100g', 'fiber_100g', 'salt_100g',
                            'fat_100g', 'carbohydrates_100g',
                            'proteins_100g', 'nutrition-score-fr_100g', 'fruits-vegetables-nuts_100g',
                            'vitamin-d_100g', 'vitamin-c_100g', 'calcium_100g', 'iron_100g']
                    st.write(df[nutcol].iloc[rec[1]])
                if param[7]:
                    try:
                        st.write('\nScore de similarité: ' + str(rec[0]))
                    except:
                        pass
        st.write('---')



