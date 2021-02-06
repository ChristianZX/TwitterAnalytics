import pandas as pd
import db_functions
import helper_functions
import time
from collections import defaultdict
from tqdm import tqdm
from sklearn.datasets import load_iris
from pure_sklearn.map import convert_estimator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


class ClfSwitcher(BaseEstimator):
    """
    A Custom BaseEstimator that can switch between classifiers.
    :param estimator: sklearn object - The classifier
    """
    def __init__(
            self,
            estimator=SGDClassifier(),
    ):

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


def build_model():
    """
    This function is only used once to determine the best model via grid search and cross validation. However this model is not used.
    Instead the best parameters are used to train a pure predict model (https://github.com/Ibotta/pure-predict) which
    is a stripped version of sklearn that inferences faster
    :return: classifier object
    """

    pipeline = Pipeline([
      #  ('vect', CountVectorizer(ngram_range=(1, 3), max_df=0.50, tokenizer=tokenize)),
      #  ('tfidf', TfidfTransformer()),
        ('clf', ClfSwitcher())
    ])

    parameters = [
        # {
        #     'clf__estimator': [MultiOutputClassifier(LinearSVC())],
        # },
        # {
        #     'clf__estimator': [MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3))],
        # },
        # {
        #     'clf__estimator': [SGDClassifier()],  # SVM if hinge loss / logreg if log loss
        #     'clf__estimator__penalty': ('l2', 'elasticnet', 'l1'),
        #     'clf__estimator__max_iter': [50, 80],
        #     'clf__estimator__tol': [1e-4],
        #     'clf__estimator__loss': ['hinge', 'log', 'modified_huber'],
        # },
        # {
        #     'clf__estimator': [MultiOutputClassifier(SGDClassifier(penalty='l2', max_iter=5, tol=1e-4, loss='hinge'))],
        # },
        #{
        #     'clf__estimator': [MultiOutputClassifier(MultinomialNB(alpha=1e-2))],
        # },
        {
            'clf__estimator': [RandomForestClassifier()],  #
            'clf__estimator__n_estimators': (100, 500, 1000, 2000),
            'clf__estimator__criterion': ['gini', 'entropy'],
            'clf__estimator__min_samples_split': [2, 4, 6],
            'clf__estimator__min_samples_leaf': [1,2,3,4],
        },
        # {
        #     'clf__estimator': [AdaBoostClassifier()],  #
        #     'clf__estimator__n_estimators': (100, 500, 1000, 2000),
        #     'clf__estimator__learning_rate': (0.001, 0.0001, 0.00001),
        #     'clf__estimator__algorithm': ['SAMME', 'SAMME.R'],
        # },
        # {
        #     'clf__estimator': [MultiOutputClassifier(svm.SVC(kernel='linear'))],
        # },
        # {
        #     'clf__estimator': [svm.SVC()],  # TBD
        #     'clf__estimator__kernel': ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
        # },
        # {
        #     'clf__estimator': [GaussianNB()], #Acc:88,4
        # },
        # {
        #     'clf__estimator': [MultinomialNB()],
        #     'clf__estimator__alpha': [1e-2, 0, 1],
        #     'clf__estimator__fit_prior': [True, False]
        # },

    ]
    classifier = GridSearchCV(pipeline, parameters, cv=2, n_jobs=6, return_train_score=False, verbose=1)
    return classifier


def create_training_matrix (load_from_db, sql_left, sql_right, clf_pure_predict_path, column_list_path):
    pickle_name_left = "df_left.pkl"
    pickle_name_right = "df_right.pkl"
    if load_from_db:
        # sql_left= """
        # select distinct u.id, u.combined_rating, u.combined_conf, f.user_id from n_followers f, n_users u
        # where cast (f.follows_ids as numeric) = u.id
        # and u.combined_conf >= 0.9
        # and u.combined_rating = 'links'
        # order by u.id
        # limit 750000
        # """
        #
        # sql_right = """
        # select distinct u.id, u.combined_rating, u.combined_conf, f.user_id from n_followers f, n_users u
        # where cast (f.follows_ids as numeric) = u.id
        # and u.combined_conf >= 0.7
        # and u.combined_rating = 'rechts'
        # order by u.id
        # limit 750000
        # """
        df_left = db_functions.select_from_db(sql_left)
        df_right = db_functions.select_from_db(sql_right)

        db_functions.save_pickle(df_left, pickle_name_left)
        db_functions.save_pickle(df_right, pickle_name_right)
    else:
        df_left= db_functions.load_pickle(pickle_name_left)
        df_right = db_functions.load_pickle(pickle_name_right)

    features = pd.concat([df_left, df_right])
    labels = features[['id', 'combined_rating']].drop_duplicates()
    del df_left
    del df_right
    features = features.pivot(index='id', columns='user_id', values='combined_rating')
    features.replace(['links', 'rechts'], 1, inplace=True)
    features.fillna(0, inplace=True)

    column_list = features.columns.values.tolist()
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    print (f"Len Features: {len(features)}")
    # classifier = build_model()
    classifier = RandomForestClassifier(n_estimators=500,
                                        criterion='entropy',
                                        min_samples_leaf=1,
                                        min_samples_split=2)
    classifier.fit(X_train, y_train['combined_rating'])

    clf_pure_predict = convert_estimator(classifier)
    db_functions.save_pickle(clf_pure_predict, clf_pure_predict_path)
    db_functions.save_pickle(classifier, "friend_rating_classifier.pkl")
    db_functions.save_pickle(column_list, column_list_path)

    predictions = classifier.predict(X_test)
    print(confusion_matrix(y_test['combined_rating'], predictions))
    print(classification_report(y_test['combined_rating'], predictions))
    print(accuracy_score(y_test['combined_rating'], predictions))


def inference_bert_friends(classifier, column_list: list, sql: str, min_matches: int):
    """
    Performs inference based on users an account follows. Stores result to n_users
    :param classifier: Classifier
    :param friend_column_list_path: Column list to be used. Any Users friends are matched against this column list
    :param sql: Sql with combination of User to be inferendes, their label (combined rating) and their friend ID
    :param min_matches: Minimum friends that must be found in friend_column for the user to get a prediction.
    More connections = more accurate prediction result
    :return:
    """
    start = time.time()
    friends = db_functions.select_from_db(sql)
    input_dataset_length = len(friends)
    print (f"SQL fetching time: {time.time() - start}")

    if len(friends) == 0:
        rows_processed = 0
        return rows_processed

    friend_set = set(friends['follows_ids'].values.tolist())
    friend_list = friends['follows_ids'].values.tolist()
    user_list = friends['user_id'].values.tolist()
    rating_list = friends['combined_rating'].values.tolist()
    del friends

    #Transforms DataFrame into DefaultDict
    relationship_dict = defaultdict(lambda: defaultdict(list))
    for i, element in enumerate (friend_list):
        relationship_dict[element][0].append(user_list[i])
        relationship_dict[element][1].append(rating_list[i])

    #conditions_not_met_string = "" #Ids in this string will still get a last seen date in DB to ignore them during next loop
    conditions_not_met_list = []  # Ids in this list will still get a last seen date in DB to ignore them during next loop
    result_dict = {}
    for element in tqdm(friend_set):
        common_friends = set(relationship_dict[element][0]) & set(column_list)
        number_of_common_friends = len(common_friends)
        if number_of_common_friends >= min_matches:
            df = pd.DataFrame(index=column_list).T
            df = df.append(pd.Series(), ignore_index=True).fillna(0)
            for friend in relationship_dict[element][0]:
                df.loc[:, friend] = 1
            df = df.iloc[:, :len(column_list)]
            prediction_proba = classifier.predict_proba(df.values.tolist())  # pure predict

            text, conf = helper_functions.conf_value("LR", prediction_proba, min_boundary = 0.5, max_boundary = 1)
            result_dict[element] = [text, conf, number_of_common_friends]
        else:
            conditions_not_met_list.append(element)

    del friend_set
    del friend_list
    del user_list
    del rating_list

    timestamp = db_functions.staging_timestamp()
    result_df = pd.DataFrame(result_dict).T
    result_df['last_seen'] = timestamp
    rows_processed = len(result_df)
    if rows_processed > 0: #checks if data has been written to DF
        db_functions.df_to_sql(result_df, "temp_table","replace")
        update_sql = """update n_users
        set bert_friends_ml_result = "0",
        bert_friends_ml_conf = cast("1" as numeric),
        bert_friends_ml_count = cast ("2" as integer),
        bert_friends_ml_last_seen = temp_table.last_seen
        from temp_table where
        cast (id as text) = temp_table."index"
        """
        start = time.time()
        db_functions.update_table(update_sql)
        db_functions.drop_table("temp_table")
        print(f"Update Time: {time.time() - start}")
    else:
        print (f"WARNING: 0 new ratings generated despite an input dataset of {input_dataset_length} rows.")

    if len (conditions_not_met_list) > 0:
        stamps = [timestamp for elm in conditions_not_met_list]
        ziped = list (zip(conditions_not_met_list,stamps))
        db_functions.df_to_sql(pd.DataFrame(ziped), "temp_table", drop = 'replace')
        sql = 'update n_users set bert_friends_ml_last_seen = temp_table."1" from temp_table where n_users.id::text = temp_table."0"'
        db_functions.update_table(sql)
        #db_functions.drop_table('temp_table')
    return rows_processed


def convert_to_pure_predict():
    #import pickle
    #from sklearn.ensemble import RandomForestClassifier
    classifier_path = "friend_rating_classifier.pkl"
    classifier = db_functions.load_pickle(classifier_path)
    clf_pure_predict = convert_estimator(classifier)
    db_functions.save_pickle(clf_pure_predict, "friend_rating_classifier_pure_predict.pkl")


def bert_friends_ml_launcher(clf_path, friend_column_list_path, sql:str, min_matches: int):
    """
    Manages Bert-Friend inference by calling the inference function until it dies not return new results anymore.
    Inference is done in batches to keep manage memory consumption
    :param clf_path: path of model the be used
    :param friend_column_list_path: Column list to be used. Any Users friends are matched against this column list
    :param sql: Sql with combination of User to be inferendes, their label (combined rating) and their friend ID
    :return: None
    """
    classifier = db_functions.load_pickle(clf_path)
    column_list = db_functions.load_pickle(friend_column_list_path)
    rows_processed = 1
    count = 0
    while rows_processed != 0:
        count += 1
        start = time.time()
        rows_processed = inference_bert_friends(classifier, column_list, sql=sql, min_matches=min_matches)
        processing_time = time.time()-start
        print (f"\nIteration: {count} | Iteration time: {processing_time} | Predictions/Second:{rows_processed/processing_time}\n | Users Updated: {rows_processed}" )

if __name__ == '__main__':
    #model_search()
    create_training_matrix(load_from_db=False)


    # clf_path = "friend_rating_classifier_pure_predict.pkl"
    # column_list_path = "friend_column_list.pkl"
    # bulk_size = 10000000
    # cool_down = 7 #accounts will only be rated every x days
    # confidence_cap_off = 0.7
    #
    # sql = f"""
    # select follows_ids, f.user_id, u.combined_rating from n_followers f, n_users u, n_users u2
    # where f.user_id = u.id
    # and cast(u2.id as text) = follows_ids
    # and u.combined_conf >= {confidence_cap_off}
	# and (u2.bert_friends_ml_last_seen is null or (substring (u2.bert_friends_ml_last_seen,0,9) > replace(((NOW() + interval '{cool_down} day')::timestamp::date::text),'-','')))
    # order by follows_ids
    # limit {bulk_size}
    # """
    #
    # min_matches = 5
    # bert_friends_ml_launcher(clf_path, column_list_path, sql=sql, min_matches=min_matches)
    print ("bing")
