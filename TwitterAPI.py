import time
from datetime import datetime, timedelta
import tweepy
import pandas as pd
from tweepy import OAuthHandler, TweepError
import API_KEYS
import db_functions
import helper_functions

API_KEY = API_KEYS.API_KEY
API_KEY_Secret = API_KEYS.API_KEY_Secret
ACCESS_TOKEN = API_KEYS.ACCESS_TOKEN
ACCESS_TOKEN_SECRET = API_KEYS.ACCESS_TOKEN_SECRET

#data_out_path =  r"C:\Analytics\CourseFiles\twitter_data_out.json"
#data_out_path2 =  r"C:\Analytics\CourseFiles\twitter_data_out_2.json"

auth = OAuthHandler(API_KEY, API_KEY_Secret)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


def api_limit():
    """
    Retrieves current API Limit from Twitter
    Commonly used API Limit Examples below:
    print (limit['resources']['followers']['/followers/ids'])
    print (limit['resources']['friends']['/friends/ids'])
    print ("['resources']['users']['/users/:id']: " + str(limit['resources']['users']['/users/:id']))
    print ("['resources']['statuses']['/statuses/show/:id']: " + str(limit['resources']['statuses']['/statuses/show/:id']))
    print ("['resources']['statuses']['/statuses/user_timeline']: " + str(limit['resources']['statuses']['/statuses/user_timeline']))
    :return:
    """
    limit = api.rate_limit_status()
    return limit



def API_Followers(screen_name: str, user_id: str, download_limit=12500000):
    """
    Downloads Follower using API. Stops after 1000 pages (5Mio Followers), no matter how many there are.
    :param screen_name: Twitter screen_name. Technically not required but added to DB for convenient lockup
    :param user_id: Twitter User_ID
    :param limit: #prevents download of more than 12,5 Million Followers of a user. Accounts with that
    many Followers are mostly non german and their followers not relevant enough to justify the download time.
    :return: none
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
            print("Scraping " + str(screen_name) + "Current Follower Count:" + str(len(followers)))
            if count >= download_limit:
                break
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

    db_functions.df_to_sql(df,'temp_followers',drop='replace')
    #db_functions.insert_to_table_followers_or_friends(df, table_name = 'n_followers', username = False)



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
    else:
        assert (method == 'search' or method == 'user_timeline' ), "Error: No known method given!"
    try:
        for fetched_tweets in tweepy.Cursor(api_method, query, count=count, tweet_mode='extended').pages(pages):
            for index in range(len(fetched_tweets)):
                id.append(fetched_tweets[index].id)
                conversation_id.append(0)
                created_at.append(str(fetched_tweets[index].created_at))
                date.append(str(fetched_tweets[index].created_at))
                try: #Checks if Tweet returns a retweet.
                    at_user = getattr(getattr(getattr(fetched_tweets[index], 'retweeted_status'), 'user'), 'screen_name')
                    tweet.append("RT @" + at_user + ": " + getattr(getattr(fetched_tweets[index], 'retweeted_status'),
                                                                   'full_text'))
                except AttributeError: #no retweet found
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
    # TODO: In case the error message is saved in variable e, print and return e?
    except TweepError as e:
        if "User not found" in str(e):
            time.sleep(20) #error are handled so fast that sleep is required to avoid rate limit exhaustian
            return "Error: User not found"
        elif "status code = 401" in str(e):
            time.sleep(20)
            print("Twitter error response: status code = 401")
            return "Twitter error response: status code = 401"
        elif "status code = 404" in str(e):
            time.sleep(20)
            print("Error 404: Account does not exist (anymore?)")
            return "Error 404: Account does not exist (anymore)"
        elif "User has been suspended" in str(e):
            time.sleep(20)
            print("Error: User has been suspended")
            return "Error: User has been suspended"
        else:
            time.sleep(20)
            print(e)
            return ("Error")

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
        print("Staging Table " + str(table_name) + " created with " + str(len(df)) + " entries.")
    else:
        return df


def tweet_details_download_launcher(table_name: str, hashtag: str, bulk_size: int = 1000, download_parent_tweets = True):
    """
    1. Calls Tweet downloader
    2. Adds details to Tweet IDs in staging table via update
    3. If Tweets are are reply to another tweet, those parent tweets can be downloaded via option download_parent_tweets
    :param table_name: Staging table name which will be updated
    :param hashtag: hashtag scraped
    :param bulk_size: number of tweets to processed in one function call
    :param download_parent_tweets: If True: Loops through all tweets until no further father elements can be found
    :return: none
    """

    parent_sql = f"""select * from {table_name} where tweet is null 
    or (retweet is not null and retweet not in (select id from {table_name})) limit {bulk_size}"""
    df_parent = db_functions.select_from_db(parent_sql)
    df_parent['id'] = df_parent['retweet'] #replaces tweet ID with parent id. Otherwise previously downloaded tweets would be downloaded again

    sql = f"select * from {table_name} where tweet is null limit {bulk_size}"
    df = db_functions.select_from_db(sql)

    download_parents = helper_functions.dataframe_length(df) == 0 and download_parent_tweets == True
    if download_parents == True:
        df = df_parent
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
            df.iloc[index:index + 1, 8:9] = result[4]  # screen_name
            df.iloc[index:index + 1, 9:10] = result[3]  # name
            df.iloc[index:index + 1, 11:12] = result[5]  # in reply to tweet id
            df.iloc[index:index + 1, 18:19] = table_name
            #print ("Fetched Tweet: {}".format(element[1]))

    if len(df) == 0:
        return 0
    db_functions.df_to_sql(df, 'temp_df', 'replace')

    #update staging table with values form temp_df
    if download_parents == False:
        sql = f"""update {table_name} a set date = b.date, tweet = b.tweet, hashtags = b.hashtags,
              user_id = cast (b.user_id as bigint), username = b.username, name = b.name,
              retweet = cast (b.retweet as bigint), staging_name = b.staging_name from (select * from temp_df) b
              where a.id = b.id"""
    else:
        sql = f"""INSERT INTO {table_name} 
        SELECT index::bigint, id, conversation_id::bigint, created_at, date, tweet, hashtags, user_id, username, name,
        link::bigint, retweet, nlikes::bigint, nreplies::bigint, nretweets::bigint, quote_url::bigint, user_rt_id::bigint,
        user_rt::bigint, staging_name FROM temp_df"""
        print (f"{new_tweets_fetched} parent tweets added.")
    db_functions.update_table(sql)
    db_functions.drop_table('temp_df')


    new_tweets_fetched = helper_functions.dataframe_length(df)
    return new_tweets_fetched


def API_get_tweet_details(tweet_id: str, sleep: bool = True):
    """
    Downloads details for single Tweet ID and returns Tweet details
    Limit Info: print(limit['resources']['statuses']['/statuses/show/:id'])
    :param tweet_id:
    :param sleep: True: Waits 0.75s to be compliant with API rate limit, False: Doesn't wait. Used for Testing only.
    :return: user_id, date, tweet text, user_name, screen_name, in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name_id
    """
    if sleep is True:
        time.sleep(0.75) # Sleep time is fitted to Twitter API download limit (900 in 15 minutes)
    try:
        tweet = api.get_status(tweet_id,tweet_mode='extended')
    except TweepError as e:
        if "Not authorized" in str(e):
            # TODO: It would be cleaner to define the text in a variable since you use it twice
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
    return tweet['user']['id'], date, tweet['full_text'], tweet['user']['name'], tweet['user']['screen_name'], \
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
        for page in tweepy.Cursor(api.friends_ids, user_id = user_id).pages():
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
        update_sql = f"update n_users set private_profile = True where id = {user_id}"
        db_functions.update_table(update_sql)
        return 0

    result_user = ["" for element in followers]
    timestamp = [datetime.now().strftime("%Y%m%d_%H%M%S") for element in followers]
    # TODO: Test if a dict of lists works here to. Instead of series.
    df = pd.DataFrame({'username': pd.Series(username_list),
                       'user_id': pd.Series(user_id_list),
                       'fav_users': pd.Series(result_user),
                       'fav_ids': pd.Series(followers),
                       'retrieve_date': pd.Series(timestamp)
                       })

    #Insert friends into table
    db_functions.insert_to_table_followers_or_friends(df, table_name = 'n_friends', username = True)
    return len(df)

def API_get_single_user_object(user_id):
    """
    Uses Api to fetch user details for given ID
    :param user_id:
    :return: screen_name, followers_count
    """
    time.sleep(0.75)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    connection = db_functions.db_connect()
    cursor = connection.cursor()
    try:
        user = api.get_user(user_id)
    except TweepError as e:
        if "User not found" in str(e):
            return "Error: User not found"
        elif "User has been suspended" in str(e):
            return "Error: User has been suspended"
        else:
            print(e)
            return (e)
    v1 = int(user['id'])
    v2 = user['name']
    v3 = user['screen_name']
    v4 = user['location']
    try:
        v5 = user['profile_location']['full_name']
    except TypeError:
        v5 = user['profile_location']
    v6 = int(user['followers_count'])
    v7 = int(user['friends_count'])
    v8 = int(user['listed_count'])
    v9 = user['created_at']
    v10 = int(user['favourites_count'])
    v11 = user['verified']
    v12 = int(user['statuses_count'])
    v13 = str(timestamp)
    user_exists_already = helper_functions.check_DB_users_existance(v1)
    if user_exists_already:
        sql = f"""update n_users set id = {v1}, name = '{v2}', screen_name = '{v3}', location = '{v4}', profile_location = '{v5}',
                followers_count = {v6}, friends_count = {v7}, listed_count = {v8}, created_at = '{v9}', favourites_count = {v10},
                verified = {v11}, statuses_count = {v12}, last_seen = '{v13}' where id = {v1}"""
        db_functions.update_table(sql)
        # sql = "INSERT INTO public.n_users(id, name, screen_name, location, profile_location, followers_count, friends_count, listed_count, created_at, favourites_count, verified, statuses_count, last_seen) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        #cursor.execute(sql)
    else:
        sql = """INSERT INTO public.n_users(id, name, screen_name, location, profile_location, followers_count,
            friends_count, listed_count, created_at, favourites_count, verified, statuses_count, last_seen)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
        cursor.execute(sql, (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13))
    connection.commit()
    cursor.close()
    print (f"{v3} Number of Followers: {v6}")
    return v3, v6