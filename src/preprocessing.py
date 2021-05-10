# ------------------------------------------ Import ------------------------------------------

# Importing the libraries
import pandas as pd
from nltk.corpus import stopwords
from functions import text_cleaning


# ------------------------------------------ Preprocessing ------------------------------------------

# Importing the dataset
listings = pd.read_csv('/home/apprenant/PycharmProjects/food_recommendation/data/openfoodfactsclean.csv')

listings['content'] = listings[['product_name']]
listings['content'] = listings['content'].astype(str).apply(text_cleaning)

list_stopwords = stopwords.words('english') + stopwords.words('french')

