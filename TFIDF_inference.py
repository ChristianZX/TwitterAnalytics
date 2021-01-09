import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import db_functions
import collections
import TFIDF_train
import helper_functions


# TODO: Might be personal preference, but I find it easier to read if everything is in the same order
#  i.e. the arguments are handled and returned in the same order as in the function defition
#  (would be really annoying to implement here though)
def load_models(pkl_filename_TFIDF: str, pkl_filename_RNDForrest: str) -> tuple:
    """
    :param pkl_filename_TFIDF: TFIDF Picklefile Path
    :param pkl_filename_RNDForrest: RND_Forest Picklefile Path
    :return: text_classifier, tfidfconverter
    """
    with open(pkl_filename_RNDForrest, 'rb') as file:
        text_classifier = pickle.load(file)

    with open(pkl_filename_TFIDF, 'rb') as file:
        tfidfconverter = pickle.load(file)

    return text_classifier, tfidfconverter


def TFIDF_inference(df: pd.DataFrame, text_classifier: RandomForestClassifier, tfidfconverter: TfidfVectorizer):
    """
    Performs Random Forrest inference
    formerly know as "TFIDF_inference_for_eval"
    :param df: dataframe with tweets that will be tested
    :param text_classifier: used classifier
    :param tfidfconverter:  used converter
    :return: prediction result
    """
    # TODO: Why rename this variable here? Parameter could simply be called df_rechts
    df_rechts = df
    lst_r_features = df_rechts.values.tolist()
    X = lst_r_features

    processed_tweets = helper_functions.scrub_tweets(X)

    tf1_new = TfidfVectorizer(max_features=5000, min_df=5, max_df=1.0, stop_words=stopwords.words('german'),
                              vocabulary=tfidfconverter.vocabulary_)
    try:
        X_tf1 = tf1_new.fit_transform(processed_tweets)
    # TODO: More precise exception?
    except:
        return [[0,0]]
    xtf2 = X_tf1.todense()  # SVM only works with dense matrix.
    try:
        predictions = text_classifier.predict(X_tf1)
    # TODO: More precise exception?
    except:
        predictions = text_classifier.predict(xtf2)

    # TODO: German variable names are bad practice
    auswertung = collections.Counter(predictions)
    return auswertung
