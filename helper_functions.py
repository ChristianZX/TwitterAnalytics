import collections
import numpy as np
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import re
from tqdm import tqdm
import db_functions
import TwitterAPI

"""common functions used by several files"""


def scrub_tweets(x):
    """
    Removes unneeded characters from tweet
    :param x: string or list of tweets
    :return: cleaned tweets
    """
    processed_tweets = []
    for tweet in range(0, len(x)):
        # Remove all the special characters
        processed_tweet = re.sub(r'\W', ' ', str(x[tweet]))
        # remove all single characters
        processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
        # Remove single characters from the start
        processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)
        # Substituting multiple spaces with single space
        processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
        # Removing prefixed 'b'
        processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
        # Converting to Lowercase
        processed_tweet = processed_tweet.lower()
        processed_tweets.append(processed_tweet)
    return processed_tweets


def lang_detect(df: pd.DataFrame) -> bool:
    """
    Analyses languages of tweets. Prints warning if >= 75% of tweets are not German
    Can either analyse dataframe or strings
    :param df: Dataframe containing column Tweets
    :return: True at least 75% of Tweets are german, else false
    """

    lang_list = []
    if isinstance(df, pd.DataFrame):
        for index, element in df.iterrows():
            try:
                lang_list.append(detect(element['tweet']))
            except LangDetectException as e:
                if "No features in text" in str(e):
                    lang_list.append("no_lang")

    if isinstance(df, str):
        try:
            lang_list.append(detect(df))
        except LangDetectException as e:
            if "No features in text" in str(e):
                lang_list.append("no_lang")

    result = collections.Counter(lang_list)
    percent_DE = result['de'] / len(lang_list) #percentage of german Tweets

    if percent_DE <= 0.50:  # checks if at least 50% of tweets are detected as german
        percent_not_german = int(100 - (100 / len(lang_list) * result['de']))
        print(f"{percent_not_german}% have been detected as non german. Account can't be evaluated meaningfully!")
        return False  # german = False

    if percent_DE <= 0.75:  # checks if at least 75% of tweets are detected as german
        percent_not_german = int(100 - round((100 / len(lang_list) * result['de']), 1))
        print(f"{percent_not_german}% have been detected as non german. Risk of a wrong result is high!")
        return False  # german = False
    else:
        return True


def interpret_stance(method: str, left, right) -> tuple:
    """
    Returns interpreation of inference result. Meant to be written to DB.
    :param method: "LR" or "pol". Pol is currently not used.
    :param left: count of left items
    :param right:  count of right items
    :return: text, conf
    """

    if method == "LR":
        if left > right:
            text = 'links'
            conf = calculate_conf(left, right)
        elif right > left:
            text = 'rechts'
            conf = calculate_conf(right, left)
        elif left == right:
            text = 'unentschieden'
            conf = 0
        else:
            print(f"Unexpected error found!: {left}, {right}")
    return text, conf


def calculate_conf(a: int, b: int) -> float:
    """
    Used by function interpret_stance to calculate stance confidence
    :param a: number of left tweets if called by left if clause, number of right tweets if called by right if clause
    :param b: number of right tweets if called by right if clause, number of left tweets if called by left if clause
    """
    conf = round(a / (a + b), 2)
    return conf


def conf_value(method: str, prediction_result: tuple, min_boundary: int = 100, max_boundary: int = 200) -> tuple:
    """
    Returns interpretation of inference result.
    :param method: "LR" or "pol"
    :param prediction_result: list with number left [0][0] and right [0][1] tweets.
    :param min_boundary: (optional) Minimum items that can be given to calculate confidence. For BERT tweet
    prediction it's always 100 (the default value)
    :param max_boundary: (optional) Maximum items that can be given to calculate confidence. For BERT tweet
    prediction it's always 200 (the default value)
    :return: text, conf
    """
    if method == "LR":
        right = prediction_result[0][1]
        left  = prediction_result[0][0]
        if right > left:
            text = 'rechts'
            conf = interpolate_max([right], right + left, min_boundary, max_boundary)
        if left > right:
            text = 'links'
            conf = interpolate_max([left], right + left, min_boundary, max_boundary)
        if left == right:
            text = 'unentschieden'
            conf = 0
    if method == "pol":
        if right > left:
            text = 'unpolitisch'
            conf = interpolate_max([right], right + left, min_boundary, max_boundary)
        if left > right:
            text = 'politisch'
            conf = interpolate_max([left], right + left, min_boundary, max_boundary)
        if left == right:
            text = 'unentschieden'
            conf = 0
    return text, conf


def interpolate_max(majority: list, number_of_predictions: int, min_boundary: int = 100,
                    max_boundary: int = 200) -> float:
    """
    Transforms rating scale so an account with 200 rated tweets, of which 100 are left gets a confidence 0%,
    200 left tweets would result in a confidence on 100%, 150 of 50% and so on...
    :param majority: Majority of tweets. If a user has 150 left tweets majority value is 150
    :param number_of_predictions: number of tweets predicted (max. 200)
    :param min_boundary: (optional) Minimum items that can be given to calculate confidence. For BERT tweet
    prediction it's always 100 (the default value)
    :param max_boundary: (optional) Maximum items that can be given to calculate confidence. For BERT tweet
    prediction it's always 200 (the default value)
    :return: confidence level

    """
    interpolated_max = np.interp(majority, [min_boundary, max_boundary], [0, number_of_predictions])
    conf = np.around(100 / number_of_predictions * interpolated_max, 2)[0]
    return conf


def calculate_combined_score(bert_friends_high_confidence_capp_off:float, self_conf_high_conf_capp_off :float,
                             min_required_bert_friend_opinions:int, user_id, self_lr:float, self_lr_conf:float, bert_friends_lr:float,
                             bert_friends_lr_conf:float, number_of_bert_friends_L:int, number_of_bert_friends_R:int):
    """
    Performs calculation of combined score
    :param bert_friends_high_confidence_capp_off: Everything above this threshold will be concidered as good result
    :param self_conf_high_conf_capp_off: good results threshold for self score
    :param min_required_bert_friend_opinions: minimum friends required to qualify as good result
    :param user_id:
    :param self_lr: self LR score
    :param self_lr_conf: self LR confidence
    :param bert_friends_lr: friend score
    :param bert_friends_lr_conf: friend score confidence
    :param number_of_bert_friends_L: number of left friends
    :param number_of_bert_friends_R: number of right friends
    :return: result, count_rated_accounts, count_rating_less_accounts,
    count_too_few_bert_friends_to_rate_and_LRself_is_invalid, count_bert_friends_result_is_mediocre,
    count_uncategorized_accounts
    """
    count_rated_accounts = 0
    count_rating_less_accounts = 0
    count_too_few_bert_friends_to_rate_and_LRself_is_invalid = 0
    count_bert_friends_result_is_mediocre = 0
    count_uncategorized_accounts = 0
    result = 0

    if self_lr == bert_friends_lr and self_lr in ['links', 'rechts'] and number_of_bert_friends_L + number_of_bert_friends_R >= min_required_bert_friend_opinions:
        new_conf = self_lr_conf * 0.65 + bert_friends_lr_conf * 0.65
        if new_conf > 1:
            new_conf = 1
        result = [self_lr, new_conf]
        count_rated_accounts = 1
    elif self_lr != bert_friends_lr:
        # Bert Friends score has high confidence, self score a mediocre confidence
        if bert_friends_lr_conf >= bert_friends_high_confidence_capp_off and number_of_bert_friends_L + \
                number_of_bert_friends_R >= min_required_bert_friend_opinions:  # high bert conf with 10 or more
            # friend opinions
            result = [bert_friends_lr, bert_friends_lr_conf]
            count_rated_accounts = 1
        # Self LR Score has high confidence, bert friend score a mediocre confidence or number of friend opinions are
        # low
        elif self_lr_conf >= self_conf_high_conf_capp_off and (
                bert_friends_lr_conf <= bert_friends_high_confidence_capp_off or number_of_bert_friends_L +
                number_of_bert_friends_R < min_required_bert_friend_opinions):
            result = [self_lr, self_lr_conf]
            count_rated_accounts = 1
        elif self_lr_conf == 0 and bert_friends_lr_conf == 0:
            count_rating_less_accounts = 1
        elif (
                self_lr_conf <= self_conf_high_conf_capp_off) and number_of_bert_friends_L + number_of_bert_friends_R \
                <= min_required_bert_friend_opinions:  # lr self rating is invalid or of low confidence and too few
            # bert friends to rate
            count_too_few_bert_friends_to_rate_and_LRself_is_invalid = 1
        elif bert_friends_lr_conf < bert_friends_high_confidence_capp_off:
            count_bert_friends_result_is_mediocre = 1
        else:
            print(user_id)
            count_uncategorized_accounts = 1
    return result, count_rated_accounts, count_rating_less_accounts, \
           count_too_few_bert_friends_to_rate_and_LRself_is_invalid, count_bert_friends_result_is_mediocre, \
           count_uncategorized_accounts


def count_friend_stances(df: pd.DataFrame, friend_lst: list, column_to_count: str,
                         min_required_bert_friend_opinions: int) -> pd.DataFrame:
    """
    Counts left and right friends in column of given dataframe
    :param df: dataframe to be used
    :param friend_lst: unique list of ID who will get a better/new rating
    :param column_to_count: column in df to count
    :param min_required_bert_friend_opinions: minimal friends a user (with an LR BERT friend score or a combined
    score) a user needs to have (more friends = better results)
    :return: Dataframe with friend stances
    """
    result_text_lst = []
    result_conf_lst = []
    result_left_count_lst = []
    result_right_count_lst = []
    result_timestamp_lst = []
    timestamp = db_functions.staging_timestamp()

    for element in tqdm(friend_lst):
        test = df[df.id == element]
        counted = test[column_to_count].value_counts()
        if counted.shape[0] > 2:
            print(f"ERROR in count_friend_stances(): got {counted.shape[0]} classes, expected two")
            assert (counted.shape[0] == 2)
        try:
            counted['links']
        except KeyError:  # no left friends
            counted['links'] = 0
        try:
            counted['rechts']
        except KeyError:  # no left friends
            counted['rechts'] = 0

        counted_values = [counted.to_list()]
        print(f"counted_values 0: {counted_values[0][0]}")
        print(f"counted_values 1: {counted_values[0][1]}")

        if counted_values[0][0] + counted_values[0][1] >= min_required_bert_friend_opinions:
            text, conf = conf_value(method='LR', prediction_result=counted_values, min_boundary=0,
                                    max_boundary=counted_values[0][0] + counted_values[0][1])
        else:
            continue
        if isinstance(conf, np.ndarray):
            conf = str(conf.squeeze())
        result_text_lst.append(text)
        result_conf_lst.append(conf)
        result_left_count_lst.append(counted_values[0][0])
        result_right_count_lst.append(counted_values[0][1])
        result_timestamp_lst.append(timestamp)

    friend_series = pd.Series(friend_lst)
    result_conf_series = pd.Series(result_conf_lst)
    result_text_series = pd.Series(result_text_lst)
    result_left_count_series = pd.Series(result_left_count_lst)
    result_right_count_series = pd.Series(result_right_count_lst)
    result_timestamp_series = pd.Series(result_timestamp_lst)

    df_result = pd.concat(
        [friend_series, result_text_series, result_conf_series, result_left_count_series, result_right_count_series,
         result_timestamp_series], axis=1)
    return df_result


def add_eval_user_tweets(moderate_ids, right_ids, table_name = 'eval_table'):
    """
    Downloads tweets of given user into table eval_table.
    :param moderate_ids: List of Twitter Account IDs with moderate stance to be added to the evaluation set
    :param right_ids: List of Twitter Account IDs with right stance to be added to the evaluation set
    :param table_name: Name of evaluation table
    :return:
    """

    sql_moderate = "select distinct cast (fh.user_id as text) from facts_hashtags fh, n_users u where fh.user_id = u.id and combined_rating in ('links') and combined_conf >= 0.75 except select distinct user_id from eval_table limit 25"
    df_moderate = db_functions.select_from_db(sql_moderate)

    sql_right = "select distinct cast (fh.user_id as text) from facts_hashtags fh, n_users u where fh.user_id = u.id and combined_rating in ('rechts') and combined_conf >= 0.75 except select distinct user_id from eval_table limit 25"
    df_right = db_functions.select_from_db(sql_right)
    combined_lst = pd.concat([df_moderate['user_id'], df_right['user_id']]).to_list()

    for element in tqdm(combined_lst):
        TwitterAPI.API_tweet_multitool(element[0], table_name, pages=1, method='user_timeline', append=True, write_to_db=True)

    print ("Bing")

def check_DB_users_existance(id):
    """
    Checks if a user is already present in table n:users
    :param id: User Id as number or string (string is preferable)
    :return: True is user exists, False if not
    """
    df = db_functions.select_from_db(f"select * from n_users where id = {id}")
    assert (len(df) <= 1),f"ERROR: Duplicate row in n_users found. Check User ID {id}"
    if len(df) == 1:
        return True
    elif len (df) == 0:
        return False


if __name__ == "__main__":
    print (check_DB_users_existance(16745492))

