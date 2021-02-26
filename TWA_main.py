import argparse
import configparser
import collections
from datetime import date
from datetime import datetime
import gc
import logging
import numpy as np
import os
import pandas as pd
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel
import sn_scrape
import sys
import time
import torch
from tqdm import tqdm
from transformers import logging
from datetime import date
import db_functions
import BERT_friends_ML
import helper_functions
from helper_functions import calculate_combined_score, count_friend_stances
import TFIDF_inference
import topic_model
import TwitterAPI




def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies and
    returns cleaned up arguments.

    :returns: arguments object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=False, help="Name or path of config file.")
    args = parser.parse_args()
    return args

def bert_predictions(tweet: pd.DataFrame, model: ClassificationModel):
    """
    Bert Inference for prediction.
    :param tweet: dataframe with tweets
    :param model: Bert Model
    :return: list of pr
    """
    tweet = tweet.values.tolist()
    try:
        predictions, raw_outputs = model.predict(tweet)
    except:
        for element in tweet.iteritems():
            model.predict([element])
        print ("STOPP")
    auswertung = collections.Counter(predictions)
    gc.collect()

    # df = pd.DataFrame(raw_outputs)
    # df['predictions'] = pd.DataFrame(predictions)
    # df['tweets'] = pd.DataFrame(tweet)
    # df = df.replace(r'\n', ' ', regex=True)
    # df_softmax = pd.DataFrame(softmax(raw_outputs, axis=1))
    # df['softmax0'] = df_softmax[0]
    # df['softmax1'] = df_softmax[1]
    # db_functions.df_to_sql(df, 'temp_table', 'replace')

    return auswertung

def init(model_path):
    """
    Loads Bert Model
    :param model_path: Path of BERT Model to load
    :return: model
    """
    os.environ['WANDB_MODE'] = 'dryrun'
    logging.set_verbosity_warning()
    train_args = {
        "reprocess_input_data": True,
        "fp16": False,
        "num_train_epochs": 30,
        "overwrite_output_dir": True,
        "save_model_every_epoch": True,
        "save_eval_checkpoints": True,
        "learning_rate": 5e-7,  # default 5e-5
        "save_steps": 5000,
        #"output_dir": output_dir,
        "warmup_steps": 2000,
        #"best_model_dir": output_dir + "/best_model/"
    }
    model = ClassificationModel("bert", model_path, num_labels=2, args=train_args)
    print(model.device)
    return model

def run():
    """
    !!!ON WINDOWS run() MUST BE CALLED FROM __main__!!!
    Details here: https://docs.python.org/2/library/multiprocessing.html#windows
    """
    torch.multiprocessing.freeze_support()
    print('loop')

def get_followers(sql, download_limit = 12500000, time_limit = False) -> None:
    """
    Is given user ids in form of SQL statement OR as List. Will retrieve followers for this user from Twitter.
    :param sql: SQL statement containing followers OR list of users IDs
    :param download_limit: max follower download for each accounts.
    :return: none
    """
    # Block 1: Check Twitter Limits
    startime = time.time()
    #timeout = 86400
    timeout = 3600
    limit = TwitterAPI.api_limit()
    ts = limit['resources']['followers']['/followers/ids']['reset']
    limit = limit['resources']['followers']['/followers/ids'][
        'remaining']  # gets the remaining limit for follower retrival
    print("Reset Time: " + str(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')))
    if limit == 0:
        print('Twitter API limit used up. Aborting query.')
        print("Reset Time: " + str(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')))
        sys.exit()
    else:
        print("Current API Follower Limit: " + str(limit))

    # Block 2: Get users whose followers we want from DB or list
    if isinstance(sql, str):
        df = db_functions.select_from_db(sql)
    elif isinstance(sql, list):
        df = pd.DataFrame(sql, columns=['id'])
    else:
        print ("Error: Must either use SQL statement or List")
        sys.exit()
    # Block 3: Get followers for each of the users retrieved in Block 2
    for index, element in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            # this option works if we load followers for cores
            id = element['user_id']
            screen_name = element['screen_name']
        except KeyError:
            # this setting is used if we load follower for anything but cores
            id = element['id']
            screen_name = 0
        print("Getting Followers of " + str(id) + " | Element " + str(index + 1) + " of " + str(len(df)))
        TwitterAPI.API_Followers(screen_name, id, download_limit = download_limit)  # <== API follower retrieval

        # Block 4: Write follower retrieve date back to n_cores
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if screen_name != 0:
            sql = "update n_cores set followers_retrieved =' " + str(timestamp) + "' where user_id = " + str(id)
            db_functions.update_table(sql)
        if startime - time.time() > timeout:
            print ("###Timeout reached!###")
            return timeout


def download_user_timelines(political_moderate_list: list, right_wing_populists_list: list):
    """
    Downloads user timelines of users featured in below lists. The downloads are used as training material for AI
    training. All lists are just incomplete examples.
    :return:
    """

    #List examples
    # political_moderate_list = ['_pik_dame_', 'Leekleinkunst', 'MartinaKraus7', 'KAFVKA', 'Volksverpetzer', 'insideX',
    #                            'FranziLucke', 'leonie_stella9', 'Ute631', 'justMPO', 'anouk_avf', 'Komisaar',
    #                            'MenschBernd', 'von_ems', 'lies_das', 'seewanda', 'Rene_Bacher', 'Prasanita93',
    #                            'IgorUllrich', 'AJSalzgitter', 'Bussi72', 'HuWutze', 'strahlgewitter', 'PhilKupi',
    #                            'BaldusEla', 'LarsKlingenberg', 'MichaelSchfer71', 'EddyAges', 'veripot', 'JoernPL',
    #                            'ondreka', 'kleinerJedi', 'DanielKinski', 'wfh7175', 'Sister_records1', 'TinaJergerkamp']
    # right_wing_populists_list = ['Junge_Freiheit', 'zaferflocken', 'HelmutWachler', 'M_Briefing', 'TinVonWo', 'mcwd12',
    #                              'EBlume3', 'h_hendrich']


    #Political unpolitical stance is currently not used
    # Tweets of below accounts will be downloaded from twitter. During model a subset of below accounts might be used.
    # unpolitical_list = ['Podolski10', 'fckoeln', 'FCBayern', 'BVB', 'rtl2', 'DMAX_TV', 'tim_kocht', 'grandcheflafer',
    #                     'bildderfrau', 'gala', 'BUNTE', 'promiflash', 'funny_catvideos', 'BibisBeauty', 'dagibee',
    #                     'siggismallz', 'Gronkh', 'CHIP_online', 'COMPUTERWOCHE', 'SkySportNewsHD', 'MOpdenhoevel',
    #                     'kayefofficial', 'VOGUE_Germany', 'lucycatofficial', 'RealLexyRoxx', 'AnselmSchindler',
    #                     'pentru_tine', 'KaJa80028344']

    #unpolitical_list = ['Podolski10'] For Testing

    # political_list = ['Thomas_Ehrhorn', 'HilseMdb', 'DirkSpaniel', 'MdB_Lucassen', 'RolandTichy', 'UllmannMdB',
    #                   'c_jung77', 'michael_g_link', 'theliberalfrank', 'IreneMihalic', 'KaiGehring', 'RenateKuenast',
    #                   'GoeringEckardt', 'MdB_Freihold', 'ZaklinNastic', 'PetraPauMaHe', 'lgbeutin', 'arnoklare',
    #                   'zierke', 'Timon_Gremmels', 'Johann_Saathoff', 'uhl_markus', 'AnjaKarliczek', 'KLeikert',
    #                   'Junge_Gruppe']

    user_lists = {'political_moderate_list': political_moderate_list,
                  'right_wing_populists_list': right_wing_populists_list}

    # List Download
    for list_name, username_list in user_lists.items():
        for element in username_list:
            TwitterAPI.API_tweet_multitool(element, list_name, pages=10, method='user_timeline',
            append=True, write_to_db=True)


def pickle_file_load_launcher(TFIDF_pol_unpol_conv, Algorithm_pol_unpol):
    """
    Loads two files from pickle:
    :param TFIDF_pol_unpol_conv: TFIDF converter
    :param Algorithm_pol_unpol: Random Forest model for Political / Unpolitical prediction
    :return: TFIDF converter, Random Forest Model
    """
    TFIDF_pol_unpol_conv, Algo_pol_unpol = TFIDF_inference.load_models(TFIDF_pol_unpol_conv, Algorithm_pol_unpol)
    return TFIDF_pol_unpol_conv, Algo_pol_unpol


def user_analyse_launcher(batch_size: int, sql: str, model_path) -> None:
    """
    Starts analysis of user in result of SQL statement. Writes results to DB in table n_users
    :param batch_size: Number of iterations to be performed before saving results to DB.
                Results are only saved to DB after all users of an iteration have been inferenced/predicted.
                Therefore it's not advisable to select a huge number of user in one iteration.
                If an error occurs during the iteration all progress will be lost.
    :param sql: SQL statement used to retrieve accounts from DB, that will be analysed.
                !!!MAKE SURE SQL STATEMENT ONLY HAS COLUMNS user_id, username!!!
    :return: none
    """

    # insert all user to n_users, that are in facts_hastags and not already in n_users (users we have seen before).
    pre_update = "insert into n_users (id) select distinct user_id from facts_hashtags except select id from n_users"
    db_functions.update_table(pre_update)

    #Unused Feature for political not political prediction
    #TFIDF_pol_unpol_conv_path = r"C:\Users\Admin\PycharmProjects\untitled\TFIDF02_pol_TFIDF_5k_SGD.pkl"  # SGD for Website
    #Algorithm_pol_unpol_path = r"C:\Users\Admin\PycharmProjects\untitled\TFIDF02_pol_SGD_5k.pkl"  # SGD for Website
    #model_path = r"F:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"

    #TFIDF_pol_unpol_conv, Algo_pol_unpol = pickle_file_load_launcher(TFIDF_pol_unpol_conv_path, Algorithm_pol_unpol_path)
    #BERT_model = inference_political_bert.load_model(model_path)
    BERT_model = init(model_path)

    # Name of temp table in DB. Is deleted at the end of this function
    table_name = 'temp_result_lr'
    #for i in tqdm(range(iterations)):  # should be enough iterations to analyse complete hashtag (Example: 5000
    # Users in Hashtag / User Batch Size 200 = 25 iterations)
    prediction_launcher(table_name, BERT_model, sql, write_to_db=True)
    #prediction_launcher(table_name, BERT_model, sql, write_to_db=True, TFIDF_pol_unpol_conv, Algo_pol_unpol)
    # analyse hashtags and write result to DB
    gc.collect()



def get_friends(sql: str):
    """
    Downloads accounts a user follows. Account IDs are given in form of an SQL statement.
    :param sql: SQL statement that delivers list of users
        Example:
        Finds distinct users of a hashtag and loads friends of all users to n_friends
        sql = "select distinct user_id from s_h_umweltsau_20201104_1540 where user_id is not null except select
        distinct user_id from n_friends"
        Example 2:
        Finds friends for users with an LR rating and a high confidence rating
        select distinct id from n_users u where lr in ('links','rechts') and lr_conf > 0.8 except select distinct
        user_id from n_friends
    :return: nothing
    """
    df = db_functions.select_from_db(sql)
    for index, element in tqdm(df.iterrows(), total=df.shape[0]):
        number_written = TwitterAPI.API_Friends(element[0], "unknown")
        print(str(number_written) + " friends written to table n_friends for id " + str(element))
        time.sleep(60)  # Avoids exceeding Twitter API rate limit
        gc.collect()  # API calls seem to be memory leaky

def friend_rating_launcher(sql: str, get_data_from_DB: bool) -> None:
    # def bert_friends_score(get_data_from_DB):
    """Refreshes score for all users in DB who...
     1) have a Bert LR rating and
     2) follow someone in n_followers
    Writes result to table n_users (total runtime 87 min)
    """
    timestamp = db_functions.staging_timestamp()
    start_time = time.time()

    if get_data_from_DB is True:
        # Runtime 18 min
        # --Bert_Friends: Zu bewertende User und die Scores ihrer Freunde
        df = db_functions.select_from_db(sql)
        db_functions.save_pickle(df, "bert_friends.pkl")
    else:
        df = db_functions.load_pickle("bert_friends.pkl")
    # df = df.iloc[:50000,:]
    df_sub0 = df.groupby(['follows_ids', 'bert_self']).size().unstack(fill_value=0)
    df_sub1 = df.groupby(['follows_ids', 'bert_friends']).size().unstack(fill_value=0)
    del df
    result = df_sub1.join(df_sub0, lsuffix='_friend_Bert', rsuffix='_self_Bert')
    del df_sub0
    del df_sub1
    user_list = result.index.to_list()
    left_friend_Bert_list = result['links_friend_Bert'].to_list()
    right_friend_Bert_list = result['rechts_friend_Bert'].to_list()
    del result

    user_dict = {}
    for i, user in enumerate(tqdm(user_list)):
        if user not in user_dict:
            user_dict[user] = {}
        right = right_friend_Bert_list[i]
        left = left_friend_Bert_list[i]
        text, conf = helper_functions.conf_value(method='LR', prediction_result=[[left, right]], min_boundary=0,
                                                 max_boundary=left + right)
        user_dict[user]["text"] = text
        user_dict[user]["confidence"] = conf
        user_dict[user]["last_seen"] = timestamp
        user_dict[user]["bf_left_number"] = left
        user_dict[user]["bf_right_number"] = right

    print ("User dict erstellt.")
    print (len (user_dict))
    result = pd.DataFrame(user_dict).T
    print("DF transponiert.")
    db_functions.df_to_sql(result, "temp_result", drop='replace')
    print("Insert into temp done.")
    sql = "update n_users set result_bert_friends = text, bert_friends_conf = cast(confidence as numeric), " \
          "bert_friends_last_seen = temp_result.last_seen, bf_left_number = temp_result.bf_left_number, " \
          "bf_right_number = temp_result.bf_right_number from temp_result where id = cast (temp_result.index as bigint)"
    db_functions.update_table(sql)
    db_functions.drop_table("temp_result")
    print(f"Runtime in  min: {(time.time() - start_time) / 60} ")


def combined_scores_calc_launcher(sql: str, bert_friends_high_confidence_capp_off, self_conf_high_conf_capp_off, min_required_bert_friend_opinions):
    """
    Calculates combined score from users self-LR score and users bert_friend score
    :return:
    """
    # limit = 1000
    # sql = f"select id, screen_name, lr, lr_conf, result_bert_friends, bert_friends_conf, bf_left_number,
    # bf_right_number from n_users where lr is not null limit {limit}"
    # sql = f"select id, screen_name, lr, lr_conf, result_bert_friends, bert_friends_conf, bf_left_number,
    # bf_right_number from n_users where lr is not null or result_bert_friends is not null"
    df = db_functions.select_from_db(sql)
    df.fillna(0, inplace=True)

    count_rated_accounts = 0
    count_uncategorized_accounts = 0
    count_rating_less_accounts = 0
    count_to_few_bert_friends_to_rate_and_LRself_is_invalid = 0
    count_bert_friends_result_is_mediocre = 0

    id_list = df['id'].to_list()
    id_dict = {i: 0 for i in id_list}

    # ToDo: Runtime 30 minutes. Changes to Dict
    for index, element in tqdm(df.iterrows(), total=df.shape[0]):
        result, rated_accounts, rating_less_accounts, to_few_bert_friends_to_rate_and_LRself_is_invalid, bert_friends_result_is_mediocre, uncategorized_accounts = calculate_combined_score(
            bert_friends_high_confidence_cap_off=bert_friends_high_confidence_capp_off,
            self_conf_high_conf_cap_off=self_conf_high_conf_capp_off,
            min_required_bert_friend_opinions=min_required_bert_friend_opinions,
            user_id=element[0],
            self_lr=element[2],
            self_lr_conf=element[3],
            bert_friends_lr=element[4],
            bert_friends_lr_conf=element[5],
            number_of_bert_friends_L=element[6],
            number_of_bert_friends_R=element[7],
            BERT_ML_rating = element[8],
            BERT_ML_conf = element[9]
        )

        id_dict[element[0]] = result
        count_rated_accounts += rated_accounts
        count_rating_less_accounts += rating_less_accounts
        count_to_few_bert_friends_to_rate_and_LRself_is_invalid += to_few_bert_friends_to_rate_and_LRself_is_invalid
        count_bert_friends_result_is_mediocre += bert_friends_result_is_mediocre
        count_uncategorized_accounts += uncategorized_accounts

    print("\n\n")
    print(f"ratingless_accounts: {count_rating_less_accounts}")
    print(
        f"to_few_bert_friends_to_rate_and_LRself_is_invalid_or_unknown_or_of_low_conf: "
        f"{count_to_few_bert_friends_to_rate_and_LRself_is_invalid}")
    print(f"bert_friends_result_is_medicore: {count_bert_friends_result_is_mediocre}")
    print(f"uncategorized_accounts: {count_uncategorized_accounts}")
    total_rating_less_accounts = count_rating_less_accounts + \
                                 count_to_few_bert_friends_to_rate_and_LRself_is_invalid + \
                                 count_bert_friends_result_is_mediocre + \
                                 count_uncategorized_accounts
    print(f"\nAccounts without rating: {total_rating_less_accounts}")
    print(f"Rated accounts: {count_rated_accounts}")
    print("\n\n")
    print("Calculation done. Writing results to DB.")

    del df

    id_dict = {k: v for k, v in id_dict.items() if v != 0}
    df_result = pd.DataFrame(id_dict).transpose()
    df_result = df_result.replace('null', np.NaN)
    if len(df_result) == 0:
        print ("Now new data.")
    else:
        print (f"{len(df_result)} new results found.")
        db_functions.df_to_sql(df_result, "temp_table", drop='replace')
        update_sql = 'update n_users set combined_rating = t."0", combined_conf = cast(t."1" as numeric) from temp_table t where n_users.id = t.index'
        db_functions.update_table(update_sql)  # runtime 8 minutes
        db_functions.drop_table("temp_table")


def get_BERT_friends_scores_from_friends(sql: str, min_required_bert_friend_opinions: int):
    """get_followers
    Counts how many left or right friends a user has. Unlike seemingly similar functions (which download and analyse
    tweets), this one gets user profiles from DB as DF.

    For users that need special attention because they did not get a combined score due to bad BERT_friend score.
    Works only if a users friends are available in n_friends.
    Afterwards get_combined_scores_from_followers(sql) needs to run again, to give the accounts handled here a
    combined score
    :param sql: sql statement that provides delivers users and their friends, which have a bert_friend or a combined
    score
    """
    df = db_functions.select_from_db(sql)

    # Accounts in this list can only rated via bert_friend_rating
    bert_friend_rating_list = df[df['combined_conf'].isnull()]['id'].drop_duplicates().to_list()
    all_ids = df['id'].drop_duplicates().to_list()
    # Account in this list can be rated via (the better) combined_rating
    combined_rating_list = [item for item in all_ids if item not in bert_friend_rating_list]

    # count left/right friends based on BERT_friend rating (widely available)
    df_result_BERT_friends = count_friend_stances(
        df,
        friend_lst=bert_friend_rating_list,
        column_to_count='result_bert_friends',
        min_required_bert_friend_opinions=min_required_bert_friend_opinions
    )

    # count left/right friends based on combined rating (better results)
    df_result_combined_ratings = count_friend_stances(
        df,
        friend_lst=combined_rating_list,
        column_to_count='combined_rating',
        min_required_bert_friend_opinions=min_required_bert_friend_opinions
    )

    df_combined = pd.concat([df_result_BERT_friends, df_result_combined_ratings])
    df_combined.dropna(inplace=True)
    del df
    del df_result_combined_ratings
    del df_result_BERT_friends

    db_functions.df_to_sql(df_combined, 'temp_scores_table', drop='replace')
    update_sql = 'update n_users set result_bert_friends = t."1", bert_friends_conf = cast (t."2" as numeric), ' \
                 'bf_left_number = t."3", bf_right_number = t."4", bert_friends_last_seen = t."5" from ' \
                 'temp_scores_table t where n_users.id = t."0"'
    db_functions.update_table(update_sql)
    db_functions.drop_table('temp_scores_table')


def prediction_launcher(table_name: str, BERT_model, sql: str,
                        write_to_db: bool = True, TFIDF_pol_unpol_conv = 0, Algo_pol_unpol = 0):
    """
    Loads 200 Tweets (one page call) per users and sends them to BERT for inference
    :param table_name: Name of temp table used to store results
    :param BERT_model: BERT Model
    :param sql: Statement providing Users to be inferenced
    :param write_to_db: True or False
    :param TFIDF_pol_unpol_conv: tfidf converter (optional)
    :param Algo_pol_unpol: Random Forrest classifier (optional)
    :return:
    """
    start_time_overal = time.time()
    cur_date = str(date.today())  # date for last seen columns
    #methods = ['pol', 'LR']
    methods = ['LR']
    data = []
    update_to_invalid_list = []

    # This DF will store all precditions results
    df_pred_data = pd.DataFrame(data,
                                columns=['user_id', 'screen_name', 'pol', 'unpol', 'pol_text', 'pol_conf', 'pol_time',
                                         'left', 'right', 'lr_text', 'lr_conf', 'lr_time', 'analyse_date'])

    sql_time_start = time.time()
    df = db_functions.select_from_db(sql)
    print(f"##############  --- SQL Select time : {sql_time_start - time.time()} --- #############")
    if df.shape[1] != 2:
        print("ERROR: DF must ONLY have columns user_id and username")
        gc.collect()
        sys.exit()
    gc.collect()


    for index, element in tqdm(df.iterrows(), total=df.shape[0]):
        start_time = time.time()
        user_id = element[0]
        screen_name = element[1]

        def tweet_download_and_lang_detect(df_tweets, user_id, update_to_invalid_list):
            """
            Calls language detection and checks if enough german tweets remain.
            If it found almost enough german Tweets it will load more.
            If it found almost none it will abort.
            :param df_tweets: 0 during first run, dataframe with tweets during later runs
            :param user_id: Twitter User_ID for tweet download and language check
            :param update_to_invalid_list: List of user that can not be downloaded from. Will append to if applicable.
            :return: df_tweets, update_to_invalid_list, abort_loop, len_df
            """
            if isinstance(df_tweets, int):
                df_tweets = TwitterAPI.API_tweet_multitool(user_id, 'temp', pages=1, method='user_timeline', append=False,
                                                           write_to_db=False)  # fills DF with 200 tweets of 1 page
                df_tweets = helper_functions.lang_detect(df_tweets)
            else:
                df_tweets_additions = TwitterAPI.API_tweet_multitool(user_id, 'temp', pages=1, method='user_timeline',
                                                           append=False,
                                                           write_to_db=False)  # fills DF with 200 tweets of 1 page
                df_tweets_additions = helper_functions.lang_detect(df_tweets_additions)
                if isinstance(df_tweets_additions, pd.DataFrame):
                    df_tweets = pd.concat([df_tweets, df_tweets_additions])
                    df_tweets.reset_index(inplace=True)
                    del df_tweets['index']

            # if df_tweets is None: #no tweets found or all tweets deleted (non german)
            #     abort_loop = True
            #     return df_tweets, update_to_invalid_list, abort_loop'

            len_df = helper_functions.dataframe_length(df_tweets)
            if len_df <= 50:
                # if almost no tweets are german don't try to get more german tweets from this users.
                # would take to many page loads
                update_to_invalid_list.append(user_id)
                abort_loop = True
            elif len_df  >= 200:
                abort_loop = True
            else:
                # if to few tweets are german load more tweets to get a better result
                abort_loop = False
            gc.collect()
            return df_tweets, update_to_invalid_list, abort_loop, len_df

        df_tweets = 0
        # tries two times to get at least 200 german tweets, if first attempt returns less than 150 german tweets
        for i in range(2):
            df_tweets, update_to_invalid_list, abort_loop, len_df = tweet_download_and_lang_detect(df_tweets, user_id, update_to_invalid_list)
            if abort_loop == True:
                break

        if len_df > 0:
            for method in methods:
                prediction_result = []
                if method == 'pol':
                    if TFIDF_pol_unpol_conv == 0 or Algo_pol_unpol == 0:
                        print ("Warning: No Political/Unpolitical classifier given. Check function parameters.")
                    else:
                        prediction_result.append(TFIDF_inference.TFIDF_inference(df_tweets['tweet'], TFIDF_pol_unpol_conv, Algo_pol_unpol))
                if method == 'LR':
                    #prediction_result.append(inference_political_bert.bert_predictions(df_tweets['tweet'], BERT_model))
                    prediction_result.append(bert_predictions(df_tweets['tweet'], BERT_model))
                runtime = int(time.time() - start_time)

                # returns text interpretation of inference
                text, conf = helper_functions.conf_value(method, prediction_result, max_boundary=len(df_tweets))

                # result and confidence score
                df_pred_data.at[index, 'user_id'] = user_id
                df_pred_data.at[index, 'screen_name'] = screen_name
                df_pred_data.at[index, 'analyse_date'] = cur_date

                # TODO: If you store the column names in variables that update depending on the method, you only need
                #  one block
                pred_result_zero = 'left' if method == "LR" else 'pol'
                pred_result_one = 'right' if method == "LR" else 'unpol'
                df_pred_data.at[index, pred_result_zero] = prediction_result[0][0]
                df_pred_data.at[index, pred_result_one] = prediction_result[0][1]

                if method == "LR":
                    df_pred_data.at[index, 'left'] = prediction_result[0][0]
                    df_pred_data.at[index, 'right'] = prediction_result[0][1]
                    df_pred_data.at[index, 'lr_text'] = text
                    df_pred_data.at[index, 'lr_conf'] = conf
                    df_pred_data.at[index, 'lr_time'] = runtime
                else:
                    df_pred_data.at[index, 'pol'] = prediction_result[0][0]
                    df_pred_data.at[index, 'unpol'] = prediction_result[0][1]
                    df_pred_data.at[index, 'pol_text'] = text
                    df_pred_data.at[index, 'pol_conf'] = conf
                    df_pred_data.at[index, 'pol_time'] = runtime

            # print("screen_name,Pol,Unpol,Pol_Time,Left,Right,LR_Time")
            # for index, element in df_pred_data.iterrows():
            #     print(
            #         f"{element['user_id']},{element['screen_name']},{element['pol']}"
            #         f",{element['unpol']},{element['pol_time']},{element['left']},{element['right']},{element['lr_time']}")
            # print("\n")
            # if index == 6:
            #     print ("Stopp")
            if (write_to_db is True and index != 0 and index % batch_size == 0) or (write_to_db is True and (df.shape[0])  == (index+1)) : #saved data x iterations OR when df has no further rows
                if len(update_to_invalid_list) > 0:
                    invalids = pd.DataFrame (update_to_invalid_list)
                    invalids['cur_date'] = cur_date
                    db_functions.df_to_sql(invalids, "temp_invalids", drop='replace')
                    update_sql = """update n_users 
                    set lr = 'invalid', pol= 'invalid', lr_pol_last_analysed = temp_invalids.cur_Date
                    from temp_invalids 
                    where id = temp_invalids."0"
                    """
                    db_functions.update_table(update_sql)
                    db_functions.drop_table("temp_invalids")

                if helper_functions.dataframe_length(df_pred_data) > 0:
                    db_functions.df_to_sql(df_pred_data, table_name, drop='replace')
                    update_sql = f"""
                                 update n_users set lr = lr_text, lr_conf = cast (a.lr_conf as numeric), pol = pol_text,
                                 pol_conf = cast (a.pol_conf as numeric), lr_pol_last_analysed = analyse_date from {table_name} 
                                 a where id = cast(user_id as bigint)"""
                    db_functions.update_table(update_sql)  # update n_users table with new resulsts

                    print(f"Data written to table: {table_name}.")
        gc.collect()
        #runtime = time.time() - start_time_overal
        #print(f"Runtime: {runtime}")
    #return df_pred_data


def eval_bert(model_path) -> None:
    """
    Runs evaluation against evaluation accounts in table "eval_table"
    Return a printout of the results
    :return: none
    """
    data = []
    df_pred_data = pd.DataFrame(data, columns=['screen_name', 'pol', 'unpol', 'pol_time', 'left', 'right', 'lr_time'])
    sql = "select distinct username from eval_table"
    #sql = "select distinct screen_name as username from n_users where id = 805308596"
    df = db_functions.select_from_db(sql)

    print("Loading BERT")
    # older version
    # model_path = r"C:\Users\Admin\PycharmProjects\untitled\outputs\political_bert_1605094936.6519241\checkpoint-15000"
    # model_path = r"F:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    model = init(model_path)
    print("Querying BERT")

    for index, element in tqdm(df.iterrows(), total=df.shape[0]):

        screen_name = element[0]

        df_tweets = TwitterAPI.API_tweet_multitool(screen_name, 'temp', pages=1, method='user_timeline', append=False,
                                                   write_to_db=False)  # speichert tweets in DF
        if isinstance(df_tweets, str):  # if df_tweets is a string it contains an error message
            continue
        start_time = time.time()
        german_language = helper_functions.lang_detect(df_tweets)
        runtime = time.time() - start_time
        print(f"Runtime Lang Detect: {runtime}")
        if german_language is False:
            continue
        start_time = time.time()
        prediction_result = [bert_predictions(df_tweets['tweet'], model)]
        runtime = time.time() - start_time
        print(f"Runtime Bert: {runtime}")

        result = prediction_result[0]
        df_pred_data.at[index, 'screen_name'] = screen_name
        try:
            df_pred_data.at[index, 'left'] = result[0]
            df_pred_data.at[index, 'right'] = result[1]
        except:
            df_pred_data.at[index, 'left'] = 0
            df_pred_data.at[index, 'right'] = 0
        df_pred_data.at[index, 'lr_time'] = runtime

    print("screen_name,Pol,Unpol,Pol_Time,Left,Right,LR_Time")
    for index, element in df_pred_data.iterrows():
        print(
            f"{element['screen_name']},{element['pol']},{element['unpol']},{element['pol_time']},{element['left']},"
            f"{element['right']},{int(element['lr_time'])}")


if __name__ == '__main__':
    args = parse_args()
    file = args.name
    print(f"Filename: {file}")
    config = configparser.ConfigParser()
    config.read(file, encoding="utf-8")

    hashtag = config['GLOBAL'].get('hashtag')

    if config['TASKS'].getboolean('download_hashtag'):
        since = config['DOWNLOAD_HASHTAG'].get('since')
        until = config['DOWNLOAD_HASHTAG'].get('until')
        download_parent_tweets = config['DOWNLOAD_HASHTAG'].getboolean('download_parent_tweets')
        sn_scrape.hashtag_download_launcher(hashtag=hashtag, since=since, until=until, download_parent_tweets = download_parent_tweets)

    if config['TASKS'].getboolean('download_friends'):
        sql = config['DOWNLOAD_FRIENDS'].get('sql')
        sql = sql.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        get_friends(sql)

    if config['TASKS'].getboolean('calculate_BERT_self_rating'):
        sql = config['CALCULATE_BERT_SELF_RATING'].get('sql')
        sql = sql.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        model_path = config['CALCULATE_BERT_SELF_RATING'].get('model_path')
        batch_size = config['CALCULATE_BERT_SELF_RATING'].getint('batch_size')
        user_analyse_launcher(batch_size=batch_size, sql=sql, model_path=model_path)

    if config['TASKS'].getboolean('calculate_LR_followers_rating'):
        sql = config['CALCULATE_LR_FOLLOWERS_RATING'].get('sql')
        sql = sql.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        get_data_from_DB = config['CALCULATE_LR_FOLLOWERS_RATING'].get('get_data_from_DB')
        friend_rating_launcher(sql, get_data_from_DB=get_data_from_DB)

    if config['TASKS'].getboolean('calculate_combined_rating'):
        bert_friends_high_confidence_cap_off = config['CALCULATE_COMBINED_RATING'].getfloat(
            'bert_friends_high_confidence_cap_off')
        self_conf_high_conf_cap_off = config['CALCULATE_COMBINED_RATING'].getfloat('self_conf_high_conf_cap_off')
        min_required_bert_friend_opinions = config['CALCULATE_COMBINED_RATING'].getint(
            'min_required_bert_friend_opinions')
        sql = config['CALCULATE_COMBINED_RATING'].get('sql')
        sql = sql.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        combined_scores_calc_launcher(sql, bert_friends_high_confidence_cap_off,
                                               self_conf_high_conf_cap_off, min_required_bert_friend_opinions)

    if config['TASKS'].getboolean('calculate_BERT_friend_rating'):
        conf_level = config['CALCULATE_BERT_FRIEND_RATING'].get('conf_level')
        min_required_bert_friend_opinions = config['CALCULATE_BERT_FRIEND_RATING'].get(
            'min_required_bert_friend_opinions')
        sql_friends = config['CALCULATE_BERT_FRIEND_RATING'].get('sql_friends')
        sql_friends = sql_friends.replace("INSERT_conf_level", conf_level)
        sql_friends = sql_friends.replace("INSERT_min_required_bert_friend_opinions",
                                          min_required_bert_friend_opinions)
        sql_friends = sql_friends.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        get_BERT_friends_scores_from_friends(sql_friends, int(min_required_bert_friend_opinions))

        bert_friends_high_confidence_cap_off = config['CALCULATE_BERT_FRIEND_RATING'].getfloat(
            'bert_friends_high_confidence_cap_off')
        self_conf_high_conf_cap_off = config['CALCULATE_BERT_FRIEND_RATING'].getfloat('self_conf_high_conf_cap_off')
        sql_combined_scores = config['CALCULATE_BERT_FRIEND_RATING'].get('sql_combined_scores')
        combined_scores_calc_launcher(sql_combined_scores, bert_friends_high_confidence_cap_off,
                                               self_conf_high_conf_cap_off, int(min_required_bert_friend_opinions))

        sql_new_results = config['CALCULATE_BERT_FRIEND_RATING'].get('sql_new_results')
        sql_new_results = sql_new_results.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        print(db_functions.select_from_db(sql_new_results))

    if config['TASKS'].getboolean('calculate_ML_friend_rating'):
        bulk_size = config['CALCULATE_ML_FRIEND_RATING'].get('bulk_size')
        cool_down = config['CALCULATE_ML_FRIEND_RATING'].get('cool_down')
        combined_conf_cap_off = config['CALCULATE_ML_FRIEND_RATING'].get('combined_conf_cap_off')
        sql = config['CALCULATE_ML_FRIEND_RATING'].get('sql')
        sql = sql.replace("INSERT_bulk_size", bulk_size)
        sql = sql.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        sql = sql.replace("INSERT_combined_conf_cap_off", combined_conf_cap_off)
        clf_path = config['CALCULATE_ML_FRIEND_RATING'].get('clf_path')
        column_list_path = config['CALCULATE_ML_FRIEND_RATING'].get('column_list_path')
        min_matches = config['CALCULATE_ML_FRIEND_RATING'].getint('min_matches')
        BERT_friends_ML.bert_friends_ml_launcher(clf_path, column_list_path, sql=sql, min_matches=min_matches)

    if config['TASKS'].getboolean('re_run_ML_friend_training'):
        load_from_db = config['RE_RUN_ML_FRIEND_TRAINING'].getboolean('load_from_db')
        sql_left = config['RE_RUN_ML_FRIEND_TRAINING'].get('sql_left')
        sql_right = config['RE_RUN_ML_FRIEND_TRAINING'].get('sql_right')
        clf_pure_predict_path = config['RE_RUN_ML_FRIEND_TRAINING'].get('clf_pure_predict_path')
        column_list_path = config['RE_RUN_ML_FRIEND_TRAINING'].get('column_list_path')
        pickle_name_left = config['RE_RUN_ML_FRIEND_TRAINING'].get('pickle_name_left')
        pickle_name_right = config['RE_RUN_ML_FRIEND_TRAINING'].get('pickle_name_right')
        classifier_pkl = config['RE_RUN_ML_FRIEND_TRAINING'].get('classifier_pkl')
        BERT_friends_ML.create_training_matrix(load_from_db, sql_left, sql_right, clf_pure_predict_path, pickle_name_left, pickle_name_right, column_list_path, classifier_pkl)

    if config['TASKS'].getboolean('download_BERT_training_data'):
        # Downloads timelines for users configured in function. Used as traning material for AI
        political_moderate_list = config['DOWNLOAD_BERT_TRAINING_DATA'].get('political_moderate_list')
        right_wing_populists_list = config['DOWNLOAD_BERT_TRAINING_DATA'].get('right_wing_populists_list')
        download_user_timelines(political_moderate_list, right_wing_populists_list)

    if config['SETUP'].getboolean('rate_eval_table_accounts'):
        # gives rating to accounts in eval_table
        model_path = config['RATE_EVAL_TABLE_ACCOUNTS'].get('model_path')
        eval_bert(model_path)

    if config['TASKS'].getboolean('show_word_cloud'):
        #plots two word clouds
        sql = config['SHOW_WORD_CLOUD'].get('sql')
        sql = sql.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        topic_model.topic_model_wordcloud(sql)

    if config['TASKS'].getboolean('download_followership'):
        user_ids = config['DOWNLOAD_FOLLOWERSHIP'].get('user_ids')
        user_ids = user_ids.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        download_limit = config['DOWNLOAD_FOLLOWERSHIP'].getint('download_limit')
        get_followers(user_ids, download_limit)
        #Insert newly donloaded followers into n_followers
        sql_insert_new_followers = config['DOWNLOAD_FOLLOWERSHIP'].get('sql_insert_new_followers')
        db_functions.update_table(sql_insert_new_followers)
        #Make sure all newly downloaded users are in table n_users
        sql_insert = config['DOWNLOAD_FOLLOWERSHIP'].get('sql_insert')
        db_functions.update_table(sql_insert)

    if config['TASKS'].getboolean('refresh_download_followership'):
        user_ids = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('user_ids')
        #user_ids = user_ids.replace("INSERT_HASHTAG", "%" + hashtag + "%")
        download_limit = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].getint('download_limit')
        get_followers(user_ids, download_limit)
        #Delete old followers from n_followers
        sql_delete_old_followers = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('sql_delete_old_followers')
        db_functions.update_table(sql_delete_old_followers)
        # Insert newly downloaded followers into n_followers
        sql_insert_new_followers = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('sql_insert_new_followers')
        db_functions.update_table(sql_insert_new_followers)
        #Make sure all newly donloaded users are in table n_users
        sql_insert = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('sql_insert')
        db_functions.update_table(sql_insert)

if config['TASKS'].getboolean('auto_run'):
    today = date.today()
    user_ids = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('user_ids')
    df = db_functions.select_from_db(user_ids)
    while True:
        # On even days refresh followers if there is something to refresh
        # Followers older than 6 months will be refreshed
        if int(today.strftime("%d")) % 2 == 0 and len(df) > 0:
            user_ids = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('user_ids')
            df = db_functions.select_from_db(user_ids)
            #user_ids = [df.iloc[0, 0]] #Only download one user at a time
            download_limit = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].getint('download_limit')
            get_followers(user_ids, download_limit)
            # Delete old followers from n_followers
            sql_delete_old_followers = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('sql_delete_old_followers')
            db_functions.update_table(sql_delete_old_followers)
            # Insert newly downloaded followers into n_followers
            sql_insert_new_followers = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('sql_insert_new_followers')
            db_functions.update_table(sql_insert_new_followers)
            # Make sure all newly donloaded users are in table n_users
            sql_insert = config['REFRESH_DOWNLOAD_FOLLOWERSHIP'].get('sql_insert')
            db_functions.update_table(sql_insert)

        #on uneven days download new followers
        else:
            user_ids = config['AUTO_RUN'].get('download_followership_user_ids')
            df = db_functions.select_from_db(user_ids)
            #user_ids = user_ids.replace("INSERT_HASHTAG", "%" + hashtag + "%")
            #user_ids = [df.iloc[0,0]]
            download_limit = config['DOWNLOAD_FOLLOWERSHIP'].getint('download_limit')
            get_followers(user_ids, download_limit)
            # Insert newly donloaded followers into n_followers
            sql_insert_new_followers = config['DOWNLOAD_FOLLOWERSHIP'].get('sql_insert_new_followers')
            db_functions.update_table(sql_insert_new_followers)
            # Make sure all newly downloaded users are in table n_users
            sql_insert = config['DOWNLOAD_FOLLOWERSHIP'].get('sql_insert')
            db_functions.update_table(sql_insert)

# Tasks?
#Add followers
#refresh followers

