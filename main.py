from datetime import datetime
import pandas as pd
import numpy as np
import twint_test
import DBA_insert
import auswertung
import twint
import TwitterAPI
import sn_scrape
from datetime import datetime, timedelta
from asyncio.exceptions import TimeoutError

def get_followers(sql):
    '''
    gets followers from a list retrieved from SQL DB
    to just retriev followers for one user use API_Followers()
    :param sql: SQL statement
    :return:
    '''
    ### SQL Statement for cores
    #sql = "select user_id, screen_name from n_cores except select user_id, username from n_followers"

    ### SQL for german users >= 10k followers

    #Block 1: Get Twitter Limits
    limit = TwitterAPI.api_limit()
    ts = limit['resources']['followers']['/followers/ids']['reset']
    limit = limit['resources']['followers']['/followers/ids']['remaining']  # gets the remaning limit for follower retrival
    print("Reset Time: " + str(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')))
    if limit == 0:
        print('Twitter API limit used up. Aborting query.')
        print("Reset Time: " + str(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')))
        try:
            cursor.close()
            DBA_insert.db_close(connection)
        except:
            print ("No cursor open")
        exit()
    else:
        print("Current API Follower Limit: " + str(limit))

    #Block 2: Get users whos followers we want from DB
    #df = DBA_insert.select_from_db("select user_id,screen_name from n_cores where followers_retrieved is null limit " + str(limit))
    df = DBA_insert.select_from_db(sql)
    #Block 3: Get followers for each of the users retrieved in Block 1
    connection = DBA_insert.db_connect()
    cursor = connection.cursor()
    for index, element in df.iterrows():
        try:
            #diese beiden sollten funktionieren, falls wir dfollowers für cores lasden
            id = element['user_id']
            screen_name = element['screen_name']
        except:
            #das hier ist die richtige einstellung, wenn wir friends für NICHT cores laden
            id = element['id']
            screen_name = 0
        print ("Getting Followers of " + str(id) + "Element " + str(index) + " of " + str(len(df)))
        #get_likes_or_follower ("followers", screen_name, id) #<== Twint folloer retrieval
        #TwitterAPI.API_Followers(screen_name, id, limit) # <== API follower retrieval
        #TwitterAPI.API_Followers(screen_name, id, 1)  # <== API follower retrieval
        TwitterAPI.API_Followers(screen_name, id, 1)  # <== API follower retrieval
    #Block 3: Write follower retrieve date back to n_cores
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sql = """update n_cores set followers_retrieved =' """ + str(timestamp) + """' where user_id = """ + str(id)
        if screen_name != 0:
            #Scrren Name ist ungleich null, wenn wir follower für CORES laden
            cursor.execute(sql)
            connection.commit()
    cursor.close()
    DBA_insert.db_close(connection)