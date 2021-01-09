import time
from datetime import datetime, timedelta
import tweepy
import pandas as pd
import numpy as np
import json
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler, TweepError
from sqlalchemy.sql import table, column, select, update, insert
from tweepy import Stream
import json
import gc
import API_KEYS
import db_functions
import API_KEYS
import db_functions

# TODO: Add API_KEYS.py file with variable names but not values, so that a new user could simply enter her
#  API key and run the program
API_KEY = API_KEYS.API_KEY
API_KEY_Secret = API_KEYS.API_KEY_Secret
ACCESS_TOKEN = API_KEYS.ACCESS_TOKEN
ACCESS_TOKEN_SECRET = API_KEYS.ACCESS_TOKEN_SECRET

#data_out_path =  r"C:\Analytics\CourseFiles\twitter_data_out.json"
#data_out_path2 =  r"C:\Analytics\CourseFiles\twitter_data_out_2.json"

auth = OAuthHandler(API_KEY, API_KEY_Secret)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


# TODO: Remove old code?
def api_limit():
    limit = api.rate_limit_status()
    #print (limit['resources']['followers']['/followers/ids'])
    #print (limit['resources']['friends']['/friends/ids'])
    #print ("['resources']['users']['/users/:id']: " + str(limit['resources']['users']['/users/:id']))
    #print ("['resources']['statuses']['/statuses/show/:id']: " + str(limit['resources']['statuses']['/statuses/show/:id']))
    #print ("['resources']['statuses']['/statuses/user_timeline']: " + str(limit['resources']['statuses']['/statuses/user_timeline']))
    return limit


# TODO: Update doc string (doesn't describe used parameters, describes an old one)
#  use only one language, preferably English
def API_Followers(screen_name: str, user_id: str):
    """
    :param screen_name: required (for SQL purpose)
    :param user_id: required
    :param remaning: [optional]: Anzahl der Läufe, die ohne sleep durchgeführt werden. Remaining = 1 for auto modus
    in which 1 minute sleep is used
    :return:
    """

    ids = []
    followers = []
    count = 0

    try:
        for page in tweepy.Cursor(api.followers_ids, user_id=user_id).pages():
            ids.extend(page)
            for index in enumerate(page["ids"]):
                followers.append(page["ids"][index[0]])
                count += 1
            time.sleep(60)
            # TODO: Is screen_name not a string?
            print("Scraping " + str(screen_name) + "Current Follower Count:" + str(len(followers)))
            #print("Lates scraping target: " + str(screen_name) + " | Current follower count: " + str(count))
    except TweepError as e:
        if "Not authorized" in str(e):
            print("Error: Not authorized. | Followers possibly set to private")
            time.sleep(60)
            return "Error: Not authorized"

    username_list = [screen_name for element in followers]
    user_id_list = [user_id for element in followers]
    result_user = ["" for element in followers]
    timestamp = [datetime.now().strftime("%Y%m%d_%H%M%S") for element in followers]
    df = pd.DataFrame({'username': pd.Series(username_list),
                       'user_id': pd.Series(user_id_list),
                       'fav_users': pd.Series(result_user),
                       'fav_ids': pd.Series(followers),
                       'retrieve_date': pd.Series(timestamp)
                       })

    connection = db_functions.db_connect()
    cursor = connection.cursor()
    
    # TODO: Generalize into function with parameters for the SQL statement and wether v1 should be run
    if screen_name == 0:
        sql = "INSERT INTO public.n_followers(user_id, follows_users, follows_ids, retrieve_date) " \
              "VALUES (%s, %s, %s, %s);"
        for index in range(len(df.index)):
            v2 = int(df.iloc[index, 1])
            v3 = (df.iloc[index, 2])
            v4 = int(df.iloc[index, 3])
            v5 = str(df.iloc[index, 4])
            cursor.execute(sql, (v2, v3, v4, v5))
            connection.commit()
    else:
        sql = "INSERT INTO public.n_followers(username, user_id, follows_users, follows_ids, retrieve_date) " \
              "VALUES (%s, %s, %s, %s, %s);"
        for index in range(len(df.index)):
            v1 = (df.iloc[index, 0])
            v2 = int(df.iloc[index, 1])
            v3 = (df.iloc[index, 2])
            v4 = int(df.iloc[index, 3])
            v5 = str(df.iloc[index, 4])
            cursor.execute(sql, (v1, v2, v3, v4, v5))
            connection.commit()
        cursor.close()
    db_functions.db_close(connection)
    print(str(len(df.index)) + " followers written to table n_followers" )
    gc.collect()


def API_tweet_multitool(query: str, table_name: str, pages: int, method: str, append: bool = False,
                        write_to_db: bool = True, count: int = 1000):
    """
    Can run twitter API calls search OR user_timeline and store result in DB table specified in input paramater
    :param query: search item e.g. hashtag
    :param table_name: name of staging table. With get a 's_h_' (for Staging_table_Hashtag) as prefix and date as suffix
    :param pages: number of cursor pages to retrieve
    :param method: API method to be used: 1.search, 2. user_timeline
    :param append: If True: Adds data to existing table. If False: replaces table table_name
    :param write_to_db: If True (default), writes data to DB, if false returns data as Dataframe
    :param count (default 1000). Number of tweets to be retrieved per user
    :return:
    """

    if append is False:
        table_name = 's_h_' + table_name+ '_' + str(db_functions.staging_timestamp())
        table_name = table_name.lower()

    #deactivates JSON which is required for api search cursor call
    api = tweepy.API(auth)

    # TODO: Multiple empty lists can be instantiated in one line, f.i. id, conversation_id = [], []
    #  or leave it be and construct a dictionary instead for easier dataframe creation (see below)
    id = []
    conversation_id = []
    created_at = []
    date = []
    tweet = []
    hashtags = []
    user_id = []
    username = []
    name = []
    link = []
    retweet = []
    nlikes = []
    nreplies = []
    nretweets = []
    quote_url = []
    user_rt_id = []
    user_rt = []

    #api_method = api.search if method == "search" else api.user_timeline
    if method == 'search':
        api_method = api.search
        wait = True
    elif method == 'user_timeline':
        api_method = api.user_timeline
        wait = False
    # TODO: Handle case where neither is True

    try:
        for fetched_tweets in tweepy.Cursor(api_method, query, count=count, tweet_mode='extended').pages(pages):
            for index in range(len(fetched_tweets)):
                id.append(fetched_tweets[index].id)
                conversation_id.append(0)
                created_at.append(str(fetched_tweets[index].created_at))
                date.append(str(fetched_tweets[index].created_at))
                tweet.append(fetched_tweets[index].full_text)
                hashtags_sub = []
                for i, element in enumerate (fetched_tweets[index].entities['hashtags']):
                    hashtags_sub.append(fetched_tweets[index].entities['hashtags'][i]['text'])
                hashtags.append(hashtags_sub)
                user_id.append(fetched_tweets[index].user.id)
                username.append(fetched_tweets[index].user.screen_name)
                name.append(fetched_tweets[index].user.name)
                link.append(0)
                retweet.append(0)
                nlikes.append(0)
                nreplies.append(0)
                nretweets.append(fetched_tweets[index].retweet_count)
                quote_url.append(0)
                user_rt_id.append(0)
                user_rt.append(0)

            if wait is True:
                time.sleep(20)
    except TweepError as e:
        if "User not found" in str(e):
            return "Error: User not found"
        elif "status code = 401" in str(e):
            print("Twitter error response: status code = 401")
            return "Twitter error response: status code = 401"
        elif "status code = 404" in str(e):
            print("Error 404: Account does not exist (anymore?)")
            return "Error 404: Account does not exist (anymore)"
        elif "User has been suspended" in str(e):
            print("Error: User has been suspended")
            return "Error: User has been suspended"
        else:
            print(e)
            return ("Error")

    # TODO: It would probably be shorter to construct the DataFrame via a dict
    #  to do this, define a dictionary with string keys and empty lists as values (instead of multiple lists)
    #  append to the lists in the dictionary
    #  convert the dictionary to a dataframe
    # Build Dataframe
    id = pd.Series(id, name='id')
    conversation_id = pd.Series(conversation_id, name='conversation_id')
    created_at = pd.Series(created_at, name='created_at')
    date = pd.Series(date, name='date')
    tweet = pd.Series(tweet, name = 'tweet')
    hashtags = pd.Series(hashtags, name='hashtags')
    user_id = pd.Series(user_id, name='user_id')
    username = pd.Series(username, name='username')
    name = pd.Series(name, name='name')
    link = pd.Series(link, name='link')
    retweet = pd.Series(retweet, name='retweet')
    nlikes = pd.Series(nlikes, name='nlikes')
    nreplies = pd.Series(nreplies, name='nreplies')
    nretweets = pd.Series(nretweets, name='nretweets')
    quote_url = pd.Series(quote_url, name='quote_url')
    user_rt_id = pd.Series(user_rt_id, name='user_rt_id')
    user_rt = pd.Series(user_rt, name='user_rt')

    df = pd.concat(
        [id, conversation_id, created_at, date, tweet, hashtags, user_id, username, name, link, retweet, nlikes,
         nreplies, nretweets, quote_url, user_rt_id, user_rt], axis=1)

    # TODO: is this still relevant?
    # if isinstance(df, pd.Series):
    #     print ("Error: 0 Tweets")
    #     return "Error: 0 Tweets"
    if df.shape[0] == 0:
        print("Error: 0 Tweets")
        return "Error: 0 Tweets"
    #Write to DB or return result
    
    if write_to_db is True:
        df = df.rename({'data-item-id': 'id', 'data-conversation-id': 'conversation_id', 'avatar': 'link'}, axis=1)
        # adds empty column to df
        df['user_id'] = ''
        # remove @ sign in username
        df['username'] = df['username'].str.replace("@", "")

        #engine, metadata = sql_alchemy_engine()  # gets SQL alchemy connection
        
        # creates list that is as long das the DF and contains tablename in every entry.
        staging_column_list = [table_name for index in range(len(df))]
        
        # appends staging_column_list to df
        df = df.assign(staging_name=pd.Series(staging_column_list).values)
        #db_functions.tweet_multitool_results_to_DB(df, table_name, append)
        db_functions.df_to_sql(df, table_name, drop = 'append')
        # TODO: Isn't table name already a string?
        print("Staging Table " + str(table_name) + " created with " + str(len(df)) + " entries.")
    else:
        return df


def tweet_details_download_launcher(table_name: str, hashtag: str, bulk_size: int = 1000):
    """
    1. Calls Tweet downloader
    2. Adds details to Tweet IDs in staging table via update
    :param table_name: Staging table name which will be updated
    :param hashtag: hashtag scraped
    :param bulk_size: number of tweets to processed in one function call
    :return: none
    """
    df = db_functions.select_from_db(f"select * from {table_name} where tweet is null limit {bulk_size}")
    for index, element in df.iterrows():
        error = False
        # downloads details for Tweets from staging table
        result = API_get_tweet_details(element[1], sleep=True)
        if result == 'Error: Not authorized': #Tweet is set to private
            error = True
        if result == 'Error: No status found with that ID.' or result == "Undefined Error":
            sql = f"update {table_name} set tweet = 'deleted' where id = {element[1]}"
            db_functions.update_table(sql)
            error = True
        if error is False:
            df.iloc[index:index + 1, 4:5] = result[1] # date
            df.iloc[index:index + 1, 5:6] = result[2] # tweet
            df.iloc[index:index + 1, 6:7] = hashtag  # hashtag
            df.iloc[index:index + 1, 7:8] = result[0]  # user_id
            df.iloc[index:index + 1, 8:9] = result[5]  # screen_name
            df.iloc[index:index + 1, 9:10] = result[4]  # name
            df.iloc[index:index + 1, 11:12] = result[6]  # in reply to tweet
            df.iloc[index:index + 1, 18:19] = table_name  # in reply to tweet
            #print ("Fetched Tweet: {}".format(element[1]))

    if len(df) == 0:
        return 0
    db_functions.df_to_sql(df, 'temp_df', 'replace')
    
    #update staging table with values form temp_df
    sql = f"update {table_name} a set date = b.date, tweet = b.tweet, hashtags = b.hashtags, " \
          "user_id = cast (b.user_id as bigint), username = b.username, name = b.name, " \
          "retweet = cast (b.retweet as bigint), staging_name = b.staging_name from (select * from temp_df) b " \
          "where a.id = b.id"
    db_functions.update_table(sql)
    db_functions.drop_table('temp_df')


# TODO: Parameter sleep is missing from docstring
def API_get_tweet_details(tweet_id: str, sleep: bool = True):
    """
    Downloads details for single Tweet ID and returns Tweet details
    Limit Info: print(limit['resources']['statuses']['/statuses/show/:id'])
    :param tweet_id:
    :return: user_id, date, tweet text, user_name, screen_name, in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name_id
    """
    if sleep is True:
        time.sleep(0.75) # Sleep time is fitted to Twitter API download limit (900 in 15 minutes)
    try:
        tweet = api.get_status(tweet_id)
    except TweepError as e:
        if "Not authorized" in str(e):
            # TODO: Technically, it would be cleaner to define the text in a variable since you use it twice
            #  or just print and return e? At this point, why are "not authorized" and "not found" handled differently?
            print("Error: Not authorized")
            return "Error: Not authorized"
        if "No status found with that ID." in str(e):
            print("Error: No status found with that ID.")
            return "Error: No status found with that ID."
        else:
            print ("Undefined Error: " + str(e.reason))
            return "Undefined Error"
    date = datetime.strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
    date = date + timedelta(hours=1) #twitter delivers GMT. +1 to get it to CET
    return tweet['user']['id'], date, tweet['text'], tweet['user']['name'], tweet['user']['screen_name'], \
           tweet['in_reply_to_status_id'], tweet['in_reply_to_user_id'], tweet['in_reply_to_screen_name']


def API_Friends(user_id: str, screen_name: str):
    """
    Calls Twitter API for friend download (users who an account follows)
    :param user_id: User_ID whose friends we want to downloadfriends
    :param screen_name: Screen name fitting to above user ID.
    :return:
    """
    ids = []
    followers = []

    #API call
    try:
        for page in tweepy.Cursor(api.friends_ids, id = user_id).pages():
            ids.extend(page)
            for index in enumerate(page["ids"]):
                followers.append(page["ids"][index[0]])
    except TweepError as e:
        # TODO: as above, 1) text is repeated three times, 2) how does it differ from e?
        if "Sorry, that page does not exist." in str(e):
            print ("Error: Sorry, that page does not exist.")
            return "Error: Sorry, that page does not exist."

    username_list = [screen_name for element in followers]
    user_id_list = [user_id for element in followers]

    #Sets user to private in DB, if tweets are not public. Avoids pulling the users another time in future
    if len(user_id_list) == 0:
        update_sql = "update n_followers set private = 1 where follows_ids = '" + str(user_id) + "'"
        db_functions.update_table(update_sql)
        return 0

    result_user = ["" for element in followers]
    timestamp = [datetime.now().strftime("%Y%m%d_%H%M%S") for element in followers]
    # TODO: Maybe test this, but the conversion to Series is likely unnecessary. A dict of lists should work just fine.
    df = pd.DataFrame({'username': pd.Series(username_list),
                       'user_id': pd.Series(user_id_list),
                       'fav_users': pd.Series(result_user),
                       'fav_ids': pd.Series(followers),
                       'retrieve_date': pd.Series(timestamp)
                       })

    #Insert friend into table
    # TODO: Use new function defined earlier that seems to do this exact thing
    connection = db_functions.db_connect()
    cursor = connection.cursor()
    sql = "INSERT INTO public.n_friends(username, user_id, follows_users, follows_ids, retrieve_date) " \
          "VALUES (%s, %s, %s, %s, %s);"
    for index in range(len(df.index)):
        v1 = (df.iloc[index, 0])
        v2 = int(df.iloc[index, 1])
        v3 = (df.iloc[index, 2])
        v4 = int(df.iloc[index, 3])
        v5 = str(df.iloc[index, 4])
        cursor.execute(sql, (v1, v2, v3, v4, v5))
        connection.commit()
    cursor.close()
    db_functions.db_close(connection)
    return len(df)
