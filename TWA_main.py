from datetime import date
import gc
import pandas as pd
import TwitterAPI
from tqdm import tqdm
import time
import db_functions
from datetime import datetime
import helper_functions
import inference_political_bert
import TFIDF_inference
import sys
import sn_scrape

from helper_functions import calculate_combined_score, count_friend_stances


def get_followers(sql: str, download_limit = 12500000) -> None:
    """
    Is given user ids in form of SQL statement. Will retrieve followers for this user from Twitter. To retrieve only followers for one user, use API_Followers()
    :param sql: SQL statement containing followers
    :param download_limit: max follower download for each accounts.
    :return: none
    """
    # Block 1: Check Twitter Limits
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

    # Block 2: Get users whose followers we want from DB
    df = db_functions.select_from_db(sql)

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


def user_analyse_launcher(iterations: int, sql: str, model_path) -> None:
    """
    Starts analysis of user in result of SQL statement. Writes results to DB in table n_users
    :param iterations: Number of times the SQL statement is used to get new users from DB.
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
    BERT_model = inference_political_bert.load_model(model_path)

    # Name of temp table in DB. Is deleted at the end of this function
    table_name = 'temp_result'
    for i in tqdm(range(iterations)):  # should be enough iterations to analyse complete hashtag (Example: 5000
        # Users in Hashtag / User Batch Size 200 = 25 iterations)
        prediction_launcher(table_name, BERT_model, sql, write_to_db=True)
        #prediction_launcher(table_name, BERT_model, sql, write_to_db=True, TFIDF_pol_unpol_conv, Algo_pol_unpol)
        # analyse hashtags and write result to DB
        update_sql = f"update n_users set lr = lr_text, lr_conf = cast (a.lr_conf as numeric), pol = pol_text, " \
                     f"pol_conf = cast (a.pol_conf as numeric), lr_pol_last_analysed = analyse_date from {table_name} " \
                     f"" \
                     f"" \
                     f"" \
                     f"a where id = cast(user_id as bigint)"
        db_functions.update_table(update_sql)  # update n_users table with new resulsts
        gc.collect()


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
        df_tweets = TwitterAPI.API_tweet_multitool(user_id, 'temp', pages=1, method='user_timeline', append=False,
                                                   write_to_db=False)  # fills DF with 200 tweets of 1 page
        if len(df_tweets) < 100:  # user has less then 100 tweets
            db_functions.update_to_invalid(cur_date, user_id)
            continue
        if isinstance(df_tweets, str):  # if df_tweets is a string it contains an error message
            db_functions.update_to_invalid(cur_date, user_id)
            continue

        german_language = helper_functions.lang_detect(df_tweets)
        if german_language is False:
            db_functions.update_to_invalid(cur_date, user_id)
            continue

        for method in methods:
            prediction_result = []
            if method == 'pol':
                if TFIDF_pol_unpol_conv == 0 or Algo_pol_unpol == 0:
                    print ("Warning: No Political/Unpolitical classifier given. Check function parameters.")
                else:
                    prediction_result.append(TFIDF_inference.TFIDF_inference(df_tweets['tweet'], TFIDF_pol_unpol_conv, Algo_pol_unpol))
            if method == 'LR':
                prediction_result.append(inference_political_bert.bert_predictions(df_tweets['tweet'], BERT_model))
            runtime = int(time.time() - start_time)

            # returns text interpretation of inference
            text, conf = helper_functions.conf_value(method, prediction_result)

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
    print("screen_name,Pol,Unpol,Pol_Time,Left,Right,LR_Time")
    for index, element in df_pred_data.iterrows():
        print(
            f"{element['user_id']},{element['screen_name']},{element['pol']}"
            f",{element['unpol']},{element['pol_time']},{element['left']},{element['right']},{element['lr_time']}")
    print("\n")

    if write_to_db is True:
        db_functions.df_to_sql(df_pred_data, table_name, drop='replace')
        print(f"Data written to table: {table_name}.")
    runtime = time.time() - start_time_overal
    print(f"Runtime: {runtime}")
    return df_pred_data


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


def refresh_score_in_DB(sql: str, get_data_from_DB: bool) -> None:
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
    result = df_sub1.join(df_sub0, lsuffix='_friend_Bert', rsuffix='_self_Bert')

    del df_sub0
    del df_sub1
    del df

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
        text, conf = helper_functions.interpret_stance("LR", left, right)
        user_dict[user]["text"] = text
        user_dict[user]["confidence"] = conf
        user_dict[user]["last_seen"] = timestamp
        user_dict[user]["bf_left_number"] = left
        user_dict[user]["bf_right_number"] = right

    result = pd.DataFrame(user_dict).T
    db_functions.df_to_sql(result, "temp_result", drop='replace')
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

    # ToDo: Runtime 90 minutes. Changes to Dict
    for index, element in tqdm(df.iterrows(), total=df.shape[0]):
        result, rated_accounts, rating_less_accounts, to_few_bert_friends_to_rate_and_LRself_is_invalid, bert_friends_result_is_mediocre, uncategorized_accounts = calculate_combined_score(
            bert_friends_high_confidence_capp_off=bert_friends_high_confidence_capp_off,
            self_conf_high_conf_capp_off=self_conf_high_conf_capp_off,
            min_required_bert_friend_opinions=min_required_bert_friend_opinions,
            user_id=element[0],
            self_lr=element[2],
            self_lr_conf=element[3],
            bert_friends_lr=element[4],
            bert_friends_lr_conf=element[5],
            number_of_bert_friends_L=element[6],
            number_of_bert_friends_R=element[7]
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
    if len(df_result) == 0:
        print ("Now new data.")
    else:
        print (f"{len(df_result)} new results found.")
        db_functions.df_to_sql(df_result, "temp_table", drop='replace')
        update_sql = 'update n_users set combined_rating = t."0", combined_conf = cast(t."1" as numeric) from temp_table t where n_users.id = t.index'
        db_functions.update_table(update_sql)  # runtime 8 minutes
        db_functions.drop_table("temp_table")


def get_BERT_friends_scores_from_friends(sql: str, min_required_bert_friend_opinions: int):
    """
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


if __name__ == '__main__':
    ###setup (not recurring)

    #Provide User_Id of followers you want to download in to table n_followers
    #sql = "select u.id from n_users u where cast (u.followers_count as bigint) >= 10000 and u.id not in (select distinct user_id from n_followers) and u.id not in (select distinct user_id from n_friends) and (u.location like '%Deutschland%' or u.location like '%Germany%')"
    #sql = "select u.id from n_users u where u.id = 760366485713879040" #refac test
    #User ID of Big Accounts, that have more then 500 followers among hashtag users
    #hashtag = 'le0711'
    #sql = "select follows_ids as id from (		select fr.follows_ids, count(fr.follows_ids)		from facts_hashtags f, n_friends fr 		where from_staging_table like '%{hashtag}%'		and f.user_id = fr.user_id		group by fr.follows_ids 		having count(fr.follows_ids) >= 500		order by count(fr.follows_ids) desc 		) a	except 	select cast (user_id as text) from n_followers"
    #get_followers(sql, download_limit = 12500000)

    # copy new users from n_followers to n_users
    # sql = "insert into n_users (id) select distinct user_id from n_followers except select id from n_users"
    # db_functions.update_table(sql)

    #Downloads timelines for users configured in function. Used as traning material for AI
    #download_user_timelines()

    #Train German Bert AI
    #train_political_bert.run_BERT_training()

    #Train Random Forrest
    #TFIDF_train.run_rnd_forrest_training()

    #evaluation based on accounts in table eval_table
    #model_path = r"F:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    #inference_political_bert.eval_bert(model_path)

    # Give LR ratings to accounts with many followers (Bert inference for big users who have not yet a self Bert score)
    #sql = "select distinct id as user_id, screen_name as username from n_followers fo, n_users u where fo.user_id = u.id and lr is null"
    #model_path = r"F:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    #user_analyse_launcher(iterations=1, sql=sql, model_path=model_path)

    ###improve L/R rated user base for better predictions
    #Give LR rating to accounts that FOLLOW many accounts that are in n_followers (Bert inference for users in n_users without self_Bert score, who follow at least 100 Big Users)
    # sql = "SELECT DISTINCT id AS user_id,                 screen_name AS username FROM n_users u, (SELECT follows_ids,          COUNT (follows_ids)    FROM n_followers f   GROUP BY follows_ids    HAVING COUNT (follows_ids) >= 100) a  WHERE CAST (a.follows_ids AS bigint) = u.id AND lr IS NULL limit 200"
    #model_path = r"F:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    #user_analyse_launcher(iterations= 300, sql, model_path=model_path)

    #Refreshes score for all users in DB who have a Bert LR rating and follow someone in n_followers
    #Since it affects all users that qualify for a LR rating, it also affects users of any downloaded hashtag you want to analyse
    # sql = "select f.follows_ids, u.screen_name, u.lr as bert_self, f.user_id, u2.lr  as bert_friends from n_users u, n_followers f, n_users u2 where u.id = cast (f.follows_ids as bigint) and u2.id = f.user_id and u2.lr in ('links','rechts') order by follows_ids"
    # refresh_score_in_DB(sql ,get_data_from_DB = True)

    ###New hashtag download (recurring)
    #Downloads entire hashtag and saves it to DB in table s_h_HASHTAG_TIMESTAMP
    # hashtag = "150JahreVaterland"
    # since = '2021-01-14'
    # until = '2021-01-20'
    # sn_scrape.hashtag_download_launcher(hashtag=hashtag, since=since, until = '2021-01-20')

    #Download friends of hashtag users, unless we already have their friends
    # hashtag = "150JahreVaterland"
    # sql = f"SELECT distinct f.user_id FROM facts_hashtags f left join n_friends fr on f.user_id = fr.user_id left join n_users u on f.user_id = u.id WHERE from_staging_table like '%{hashtag}%' and fr.user_id is null and u.private_profile is null"
    # get_friends(sql)

    #Give Bert LR-self rating to all new users in hashtag
    #The LR-self rating is based on the users tweets
    hashtag = "150JahreVaterland"
    #hashtag = 'le0711'
    #hashtag = 'b2908'
    #To Do: improve Long SQL runtime
    sql = f"""
    SELECT DISTINCT u.id AS user_id, screen_name AS username
    FROM n_users u, facts_hashtags fh,
    (SELECT follows_ids,COUNT (follows_ids) FROM n_followers f GROUP BY follows_ids HAVING COUNT (follows_ids) >= 100) a
    WHERE CAST (a.follows_ids AS bigint) = u.id
    AND lr IS NULL
    AND fh.user_id = u.id
    AND fh.from_staging_table like '%{hashtag}%'
    --limit 200
    """
    model_path = r"F:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    user_analyse_launcher(iterations=1, sql=sql, model_path=model_path)

    # Overwrite / Refresh LR-self rating for all users of a hashtag
    #sql = f"select distinct u.id AS user_id, screen_name AS username  from facts_hashtags f, n_users u where f.hashtags = '{hashtag}' and f.user_id = u.id and u.lr is null"
    #user_analyse_launcher(iterations= 1, sql=sql, model_path=model_path)

    #Refreshes LR-friend rating for all users in hashtag who have a Bert LR rating and follow someone in n_followers
    #The LR-friend rating is based on the LR-self rating of account a user follows
    #hashtag = 'le0711'
    # hashtag = 'b2908'
    # sql = f"""
    # SELECT DISTINCT f.follows_ids, u.screen_name, u.lr AS bert_self, f.user_id, u2.lr AS bert_friends
    # FROM n_users u, n_followers f, n_users u2, facts_hashtags fh
    # WHERE u.id = CAST (f.follows_ids AS bigint)
    # AND u2.id = f.user_id
    # AND u2.lr in ('links',
    # 'rechts')
    # AND fh.user_id = u.id
    # AND fh.from_staging_table like '%{hashtag}%'
    # ORDER BY follows_ids
    # """
    # refresh_score_in_DB(sql ,get_data_from_DB = True)

    #Calculates combined score from users self-LR score and users bert_friend score
    #all users
    # bert_friends_high_confidence_capp_off = 0.70
    # self_conf_high_conf_capp_off = 0.70
    # min_required_bert_friend_opinions = 10
    # sql = """
    # SELECT id, screen_name, lr, lr_conf, result_bert_friends, bert_friends_conf, bf_left_number, bf_right_number
    # FROM n_users
    # WHERE (lr IS NOT NULL OR result_bert_friends IS NOT NULL)
    # AND combined_rating IS NULL
    # """
    #hashtag only
    #hashtag = 'le0711'
    #hashtag = 'b2908'
    # sql = f"""
    # SELECT id, screen_name, lr, lr_conf, result_bert_friends, bert_friends_conf, bf_left_number, bf_right_number
    # FROM n_users
    # WHERE (lr IS NOT NULL OR result_bert_friends IS NOT NULL)
    # AND combined_rating IS NULL
    # AND id in
    #    (SELECT DISTINCT f.user_id
    #     FROM facts_hashtags f
    #     WHERE from_staging_table like '%{hashtag}%')
    # """
    #test_sql = "SELECT id,     screen_name,       lr,       lr_conf,       result_bert_friends,       bert_friends_conf,       bf_left_number,       bf_right_number  from n_users where id = 843210563190771713"
    # combined_scores_calc_launcher(sql, bert_friends_high_confidence_capp_off, self_conf_high_conf_capp_off, min_required_bert_friend_opinions)

    # Counts how many left or right friends a user has. Unlike seamingly similar functions (who download and analyse tweets),
    # this one gets user profiles from DB as DF.
    # For users that need special attention because they did not get a combined score due to bad BERT_friend score.
    # hashtag = 'le0711'
    # hashtag = 'b2908'
    # conf_level = 0.70
    # min_required_bert_friend_opinions = 10
    # sql = f"""
    # SELECT DISTINCT u.id, u2.id AS friend_id, u2.lr, u2.lr_conf, u2.result_bert_friends, u2.bert_friends_conf,
    # u2.bf_left_number, u2.bf_right_number, u2.combined_rating, u2.combined_conf
    # FROM facts_hashtags f, n_users u, n_friends fr, n_users u2
    # WHERE from_staging_table like '%{hashtag}%'
    # AND f.user_id = fr.user_id
    # AND fr.user_id = u.id
    # AND CAST (fr.follows_ids AS numeric) = u2.id
    # AND u.combined_rating IS NULL
    # AND u2.result_bert_friends in ('links', 'rechts')
    # AND u2.bert_friends_conf >= {conf_level}
    # ORDER BY u.id
    # """
    # get_BERT_friends_scores_from_friends(sql, min_required_bert_friend_opinions)

    #Run combined_scores for users that got a friend rating after downloading their friends directly
    # hashtag = 'le0711'
    # hashtag = 'b2908'
    # sql = f"""
    # SELECT id, screen_name, lr, lr_conf, result_bert_friends, bert_friends_conf, bf_left_number, bf_right_number
    # FROM n_users
    # WHERE (lr IS NOT NULL OR result_bert_friends IS NOT NULL)
    # AND combined_rating IS NULL
    # AND id in
    #     (SELECT DISTINCT f.user_id
    #      FROM facts_hashtags f
    #      WHERE from_staging_table like '%{hashtag}%')"""
    #
    # combined_scores_calc_launcher(sql, bert_friends_high_confidence_capp_off, self_conf_high_conf_capp_off,
    #                               min_required_bert_friend_opinions)

    #Print number of newly found results
    #hashtag = 'le0711'
    #hashtag = 'b2908'
    # sql = f"""
    # SELECT COUNT (DISTINCT u.id)
    # FROM facts_hashtags f, n_users u
    # WHERE from_staging_table like '%{hashtag}%'
    #   AND f.user_id = u.id
    #   AND combined_rating in ('links', 'rechts')"""
    #
    # print (db_functions.select_from_db(sql))