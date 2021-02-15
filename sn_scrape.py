import os
import pandas as pd
import db_functions
import TwitterAPI
import time
from tqdm import tqdm


def SN_get_tweet_ids_for_hashtag(since: str, until: str, hashtag: str) -> list:
    """
    Uses snScrape to download Tweet IDs of a Hashtag
    :return: list of tweet_ids
    """
    stream_line = f'snscrape twitter-search "#{hashtag} since:{since} until:{until}"'
    stream = os.popen(stream_line)
    output = stream.readlines()
    cleaned_tweet_ids = []
    for element in output:
        temp = element.strip("\n")
        cleaned_tweet_ids.append(temp[temp.rfind("/") + 1:])
    return cleaned_tweet_ids


def SN_db_operations(hashtag: str, since: str, until: str) -> tuple:
    """
    1. Creates staging table for new hashtag
    2. Calls tweet ID downloader
    3. Saves tweet IDs to DB
    Downloads TweetIDs of Hashtags via snScrapeTweet
    :param hashtag: hashtag, to be downloaded
    :param since: start date for tweet download (string)
    :param until: until date for tweet download (string)
    :return: table_name, hashtag,  dataframe length
    """

    table_name = db_functions.get_staging_table_name(hashtag) #adds prefix (s_h_) and suffix (dateand time) to tablename
    db_functions.drop_table(table_name)
    db_functions.create_empty_staging_table(table_name)
    tweet_ids = SN_get_tweet_ids_for_hashtag(since, until, hashtag) #downloads tweets
    df = pd.DataFrame(tweet_ids)
    write_to_table = "update_temp"

    db_functions.df_to_sql(df, write_to_table, drop="replace") #Write Tweet ID to temp table
    #db_functions.update_table('insert into ' + str(table_name) + ' (id) select cast ("0" as bigint) from update_temp') #insert temp table content into staging table
    db_functions.update_table(f'insert into {table_name} (id) select cast ("0" as bigint) from {write_to_table}')  # insert temp table content into staging table


    try:
        print(f"Table {table_name} created with {len(df)} tweets.")
    except:
        print("Unhandled Error")
    db_functions.drop_table(write_to_table)
    return table_name, hashtag, df.shape[0]


def hashtag_download_launcher(hashtag, since: str, until: str, download_parent_tweets: bool):
    """
    Manages whole hashtag download process.
    Step1: Calls procedures for SQL table creation, Tweet ID download
    Step2: Calls Tweet Details download launcher to add tweet details. Does this in iterations to minimize data loss risk
    Result is stored in staging table s_h_HASHTAG_TIMESTAMP
    :param hashtag: hashtag, to be downloaded
    :param since: start date for tweet download (string)
    :param until: until date for tweet download (string)
    :return: table_name
    """
    table_name, hashtag, len_df = SN_db_operations(hashtag, since, until)
    print("Hashtag Twitter ID download complete. Starting detail download.")
    bulk_size = 1000



    new_tweets_fetched = 1
    loop_counter = 1
    while new_tweets_fetched != 0:
        start_time = time.time()
        print (f"Iteration {loop_counter} running. Estimated Iterations {int (len_df / bulk_size) +1+5}")
        #+1 to avoid 0 itteratios, +5 is a good estimate for number of iterations to get all parent tweets
        new_tweets_fetched = TwitterAPI.tweet_details_download_launcher(table_name, hashtag, bulk_size, download_parent_tweets)
        loop_counter += 1
        print (f"Iteration {loop_counter} runtime: {time.time() - start_time} ")
    print("Hashtag downloaded successfully.")

    #insert new users into table n_users
    new_users = f"""insert into n_users (id)
    select f.user_id from {table_name} f  left join n_users u on f.user_id = u.id
    where u.id is null and f.user_id is not null"""
    db_functions.update_table(new_users)
    return table_name

