import os
import pandas as pd
import db_functions
import TwitterAPI
from tqdm import tqdm


# TODO: documentation in English?
def SN_get_tweet_ids_for_hashtag(since: str, until: str, hashtag: str) -> list:
    """
    Nutzt snScrape um die tweet IDs in einem Hashtag herunterzuladen
    :return: list of tweet_ids
    """
    # stream = os.popen('snscrape twitter-search "#umweltsau since:2019-09-01 until:2019-10-15"')
    stream_line = f'snscrape twitter-search "#{hashtag} since:{since} until:{until}"'
    stream = os.popen(stream_line)
    output = stream.readlines()
    cleaned_tweet_ids = []
    for element in output:
        temp = element.strip("\n")
        cleaned_tweet_ids.append(temp[temp.rfind("/") + 1:])
    return cleaned_tweet_ids


# TODO: describe return value in doc string
def SN_db_operations(hashtag: str, since: str, until: str) -> tuple:
    """
    1. Creates staging table for new hashtag
    2. Calls tweet ID downloader
    3. Saves tweet IDs to DB
    Downloads TweetIDs of Hashtags via snScrapeTweet
    :param hashtag: hashtag, to be downloaded
    :param since: start date for tweet download (string)
    :param until: until date for tweet download (string)
    :return:
    """
    table_name = db_functions.get_staging_table_name(hashtag) #adds prefix (s_h_) and suffix (dateand time) to tablename
    db_functions.drop_table(table_name)
    db_functions.create_empty_staging_table(table_name)
    tweet_ids = SN_get_tweet_ids_for_hashtag(since, until, hashtag) #downloads tweets
    df = pd.DataFrame(tweet_ids)

    # TODO: as the name is used three times, it would be good practice to save "update_temp" in a variable
    db_functions.df_to_sql(df, "update_temp", drop="replace") #Write Tweet ID to temp table
    # TODO: table_name should already be a string?
    db_functions.update_table('insert into ' + str(table_name) + ' (id) select cast ("0" as bigint) from update_temp') #insert temp table content into staging table
    try:
        print(f"Table {table_name} created with {len(df)} tweets.")
    except:
        print("Unhandled Error")
    db_functions.drop_table("update_temp")
    return table_name, hashtag, df.shape[0]


def hashtag_download_launcher(hashtag, since: str, until: str):
    """
    Manages whole hashtag download process.
    Step1: Calls procedures for SQL table creation, Tweet ID download
    Step2: Calls Tweet Details download launcher to add tweet details. Does this in iterations to minimize data loss risk
    Result is stored in staging table s_h_HASHTAG_TIMESTAMP
    :param hashtag: hashtag, to be downloaded
    :param since: start date for tweet download (string)
    :param until: until date for tweet download (string)
    :return: none
    """
    # TODO: Use function parameters instead of hardcoded values
    table_name, hashtag, len_df = SN_db_operations('le0711', '2019-12-01', '2020-12-03')
    print("Hashtag Twitter ID download complete. Starting detail download.")
    bulk_size = 1000
    for i in tqdm(range(int(len_df / bulk_size + 1))):
        TwitterAPI.tweet_details_download_launcher(table_name, hashtag, bulk_size)
    print("Hashtag downloaded successfully.")


# TODO: Is main still relevant?
if __name__ == '__main__':
    #Downloads entire hashtag and saves it to DB in table s_h_HASHTAG_TIMESTAMP
    #hashtag_download_launcher(hashtag='le0711', since='2019-12-01', until='2020-12-03')
    pass
