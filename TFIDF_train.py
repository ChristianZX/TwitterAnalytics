import time
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import db_functions
from langdetect import detect
import collections
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import inference_political_bert
import helper_functions

def delete_non_german_tweets_from_df(df):
    """
    #deletes non german tweets from training dataframe
    :param df:
    :return: df
    """
    for df_index, df_element in tqdm(df.iterrows(),total=df.shape[0]):
        german_language = helper_functions.lang_detect(df_element['tweet'], update=True)
        if german_language == False:
            df = df.drop(([df_index]), axis=0)
    return df

def run_rnd_forrest_training():
    """
    Rund Random Forrest training. Stores model in pickle file
    :return: none
    """
    start_time = time.time()
    pkl_filename_converter = "TFIDF02_pol_TFIDF.pkl"
    pkl_filename_classifier = "TFIDF02_pol_RND_Forrest.pkl"

    sql_politisch = "select screen_name, tweet from temp_politische_tweets_2k where pol = 1 union select username as screen_name, tweet from politische_tweets union select username as screen_name, tweet from  facts_hashtags where from_staging_table like '%sturm%'"
    sql_unpolitical = "select screen_name, tweet from temp_unpolitische_tweets_2k where unpol = 1"
    df_unpolitical = db_functions.select_from_db(sql_unpolitical)
    df_unpolitical = df_unpolitical.assign(label=1)
    df_unpolitical = delete_non_german_tweets_from_df(df_unpolitical) #deletes non german tweets from training dataframe
    df_political = db_functions.select_from_db(sql_politisch)
    df_political = df_political.assign(label=0)
    df_political = delete_non_german_tweets_from_df(df_political)

    df = pd.concat([df_unpolitical,df_political])
    df['tweet'] = df['tweet'].str.replace('\r', "")
    df = df.sample(frac=1)
    X = df['tweet'].tolist()
    y = df ['label'].tolist()

    processed_tweets = helper_functions.scrub_tweets(X)

    tfidfconverter = TfidfVectorizer(max_features=5000, min_df=5, max_df=1.0)
    X = tfidfconverter.fit_transform(processed_tweets).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #Random Forest Classifier
    text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    text_classifier.fit(X_train, y_train)

    # Save to file in the current working directory
    with open(pkl_filename_converter, 'wb') as file:
        pickle.dump(tfidfconverter, file)

    with open(pkl_filename_classifier, 'wb') as file:
        pickle.dump(text_classifier, file)

    predictions = text_classifier.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))

    print ("\n")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    run_rnd_forrest_training()
