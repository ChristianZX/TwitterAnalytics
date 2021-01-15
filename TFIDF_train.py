import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import pickle
import db_functions
import helper_functions


def delete_non_german_tweets_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deletes non-German tweets from training dataframe
    :param df: dataframe containing tweets of all languages
    :return: df containing only tweets detected as german
    """
    for df_index, df_element in tqdm(df.iterrows(), total=df.shape[0]):
        german_language = helper_functions.lang_detect(df_element['tweet'], update=True)
        if german_language is False:
            df = df.drop(([df_index]), axis=0)
    return df


def run_rnd_forrest_training(sql_political, sql_unpolitical, pkl_filename_converter, pkl_filename_classifier) -> None:
    """
    Run Random Forrest training. Stores model in pickle file.
    :param sql_political: Political Tweets selected from DB
    :param sql_unpolitical: Unpolitical Tweets selected from DB
    :param pkl_filename_converter: name of TFIDF converter pickle file
    :param pkl_filename_classifier: name of classifier pickle file
    :return: none
    """
    start_time = time.time()
    df_unpolitical = db_functions.select_from_db(sql_unpolitical)
    df_unpolitical = df_unpolitical.assign(label=1)
    df_unpolitical = delete_non_german_tweets_from_df(df_unpolitical)
    
    df_political = db_functions.select_from_db(sql_political)
    df_political = df_political.assign(label=0)
    df_political = delete_non_german_tweets_from_df(df_political)
    
    df = pd.concat([df_unpolitical, df_political])
    df['tweet'] = df['tweet'].str.replace('\r', "")
    df = df.sample(frac=1)
    X = df['tweet'].tolist()
    y = df['label'].tolist()
    processed_tweets = helper_functions.scrub_tweets(X)
    
    tfidfconverter = TfidfVectorizer(max_features=5000, min_df=5, max_df=1.0)
    X = tfidfconverter.fit_transform(processed_tweets).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Random Forest Classifier
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
    print("\n")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    pkl_filename_converter = "TFIDF02_pol_TFIDF.pkl"
    pkl_filename_classifier = "TFIDF02_pol_RND_Forrest.pkl"
    sql_political = "select screen_name, tweet from temp_politische_tweets_2k where pol = 1 union select username as " \
                    "screen_name, tweet from politische_tweets union select username as screen_name, tweet from  " \
                    "facts_hashtags where from_staging_table like '%sturm%'"
    sql_unpolitical = "select screen_name, tweet from temp_unpolitische_tweets_2k where unpol = 1"
    run_rnd_forrest_training(sql_political, sql_unpolitical, pkl_filename_converter, pkl_filename_classifier)
