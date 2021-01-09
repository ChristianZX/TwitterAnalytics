import warnings
import re
import math
from datetime import datetime, timedelta
from datetime import timezone
import warnings
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
from decimal import Decimal, ROUND_UP
from sqlalchemy import create_engine, Table, MetaData
import sys
from tqdm import tqdm
import numpy as np
import TwitterAPI
import pickle
import API_KEYS

# v.1.0

def df_to_sql(df, tablename, drop=False):
    """
    Writes Dataframe to DB
    :param df: name of dataframe that will be written to DB
    :param tablename: tablename for DB
    :param drop: 'replace' , 'append' or False to raise error when table already exists
    """
    engine, metadata = sql_alchemy_engine()
    if drop == "replace":
        drop_table(tablename, engine, metadata)
        df.to_sql(tablename, con=engine)
    elif drop == False:
        df.to_sql(tablename, con=engine)
    elif drop == "append":
        df.to_sql(tablename, con=engine, index = False, if_exists='append')
    else:
        print ("Error raised by df_to_sql(): None of the write conditions (replace, drop, append) was met. Check function parameters!")
        sys.exit()
    engine.dispose

# def tweet_multitool_results_to_DB(df, tablename, append=False):
#     '''
#     CAN ONLY BE USED BY TwitterAPY.API_tweet_multitool! For general purposes use db_functions.df_to_sql()
#     adds staging table column name and writes df_sub to database
#     :param df: dataframe
#     :param tablename: tablename to be created in DB
#     :param append: replace existing table if false
#     '''
#
#     df = df.rename({'data-item-id': 'id', 'data-conversation-id': 'conversation_id', 'avatar': 'link'}, axis = 1)
#     #adds empty column to df
#     df['user_id'] = ''
#     #remove @ sign in username
#     df['username'] = df['username'].str.replace("@", "")
#
#     engine, metadata = sql_alchemy_engine() #gets SQL alchemy connection
#     staging_column_list = [tablename for index in range(len(df))] #creates list that is as long das the DF and contains tablename in every entry.
#     df = df.assign(staging_name=pd.Series(staging_column_list).values) #appends staging_column_list to df
#
#     if append == False:
#         #Drops staging table if already existing
#         df.to_sql(tablename, con=engine, if_exists='replace')  # writes df to Database using SQL alchemy engine
#     else:
#         #Appends to existing table
#         df.to_sql(tablename, con=engine, if_exists='append')  # writes df to Database using SQL alchemy engine


def init_db_connections():
    """
    Convienience function so I dont have to call sql_alchemy_engine() and  db_connect() separately
    :return: engine, connection, metadata
    """
    engine, metadata = sql_alchemy_engine()
    connection = db_connect()
    return engine, connection, metadata


def sql_alchemy_engine():
    '''
    Replace engine_sting with code following below example
    engine = create_engine('postgresql://postgres:YOUR_PASSWORD@localhost:5433/tw_lytics01')
    :return: engine, metadata
    '''
    engine = create_engine(API_KEYS.engine_string)
    metadata = MetaData(engine)
    return engine, metadata


def staging_timestamp():
    """
    Creates Timestamp in format "%Y%m%d_%H%M"
    :return: timestamp
    """
    my_timestamp = datetime.now()
    my_timestamp = my_timestamp.strftime("%Y%m%d_%H%M")
    return my_timestamp


def db_connect():
    '''
    Replace connection_string with code following below example:
        connection = psycopg2.connect(user="postgres",
                                  password="YOUR_PASSWORD",
                                  host="127.0.0.1",
                                  port="5433",
                                  database="YOUR_DB_NAME")
    Hint: often it's port 5432
    :return: psycopg2 connection
    '''
    try:

        connection = API_KEYS.connection_string
        cursor = connection.cursor()
        # Print PostgreSQL Connection properties
        # print ( connection.get_dsn_parameters(),"\n")

        # Print PostgreSQL version
        cursor.execute("SELECT version();")
        record = cursor.fetchone()
        # print("You are connected to - ", record,"\n")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", str(error))
    return connection


def db_close(connection):
    try:
        # closing database connection.
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    except:
        pass
        # print ("It seems no DB connection not was open.")


def drop_table(name, engine=0, metadata=0):
    """
    :param name: Table name to be dropped
    :param engine: optional DB engine (to be removed in next version)
    :param metadata: optional DB metadata (to be removed in next version)
    """
    engine, metadata = sql_alchemy_engine()
    table_to_drop = Table(name, metadata)
    table_to_drop.drop(engine, checkfirst=True)

    # df_5.to_sql('isert_test01', con=engine)  # writes df to Database using SQL alchemy engine


def select_from_db(sql):
    '''
    Sends an SQL Statement to the DB
    :param sql: SQL Statement that will be executed by the DB
    :return: Dataframe with Query Result
    '''
    connection = db_connect()
    df = sqlio.read_sql_query(sql, connection)  # stagin table
    db_close(connection)
    return df


def update_table(sql):
    connection = db_connect()
    cursor = connection.cursor()
    print (sql)
    cursor.execute(sql)
    connection.commit()
    cursor.close()
    db_close(connection)
    print("Update statement send: " + sql)

def update_to_invalid(cur_date, user_id):
    """
    used to set columns to invalid in n_users. These columns are not used anymore during next run. Used for non german tweets and tweets of deletes users
    :return:
    """
    sql_invalid = f"update n_users set lr = 'invalid', pol = 'invalid', lr_pol_last_analysed = '{cur_date}' where id  = {user_id}"
    update_table(sql_invalid)
    print(f"Set {user_id} to invalid")


def get_staging_table_name(table_name):
    return 's_h_' + table_name + '_' + str(staging_timestamp())

def create_empty_staging_table(table_name):
    sql = 'CREATE TABLE {} (   index bigint,    id bigint,    conversation_id bigint,    created_at text COLLATE pg_catalog."default",    date text COLLATE pg_catalog."default",     tweet text COLLATE pg_catalog."default",    hashtags text COLLATE pg_catalog."default",    user_id bigint,    username text COLLATE pg_catalog."default",    name text COLLATE pg_catalog."default",    link bigint,    retweet bigint,    nlikes bigint,    nreplies bigint,    nretweets bigint,    quote_url bigint,    user_rt_id bigint,    user_rt bigint,    staging_name text COLLATE pg_catalog."default")'.format(table_name)
    update_table(sql)

# def insert_new_tweets_into(source_table, connection, target_table='facts'):
#     '''
#     fügt NEUE Twitter Daten den alten hinzu, dropt die bisherige Tabelle und ersetzt sie durch die concatinierten Datamarts
#     Komplexe implementierung aber schnell
#     :param source_table: Name der Staging Tabelle
#     :param connection: Name der psycopg2 connection
#     :param target_table: Tabelle in die geschrieben wird. Default Wert 'facts'
#     :return: nothing
#     '''
#     start_time = datetime.now()
#
#     sql = "SELECT * FROM public." + source_table
#     df_source = sqlio.read_sql_query(sql, connection)  # stagin table
#     sql = "SELECT * FROM public." + target_table
#     df_target = sqlio.read_sql_query(sql, connection)  # fact table
#     df_duplicates = df_source[df_source.id.isin(
#         df_target.id) == True]  # findet die indicies, die schon in der Faktentabelle sind. Ich will prüfen, ob die aus einer Staging Tablle...
#     # mit dem selben Hashtag kommen wie das, nachdem gerade gesucht wird. Das brauche ich, um entscheiden zu können ob es eine Datenlücke gibt oder nicht.
#     df = df_source[
#         df_source.id.isin(df_target.id) == False]  # findet indices die noch nicht in der Faktentabelle enthalten sind
#     cursor = connection.cursor()
#     # cursor.execute("INSERT INTO a_table (c1, c2, c3) VALUES(%s, %s, %s)", (v1, v2, v3))
#     for index in range(len(df.index)):
#         v1 = int(df.iloc[index, 0])
#         v2 = int(df.iloc[index, 1])
#         v3 = int(df.iloc[index, 2])
#         v4 = int(df.iloc[index, 3])
#         v5 = df.iloc[index, 4]
#         v5 = datetime.strptime(v5, '%Y-%m-%d %H:%M:%S')
#         v6 = df.iloc[index, 5]
#         v7 = df.iloc[index, 6]
#         v8 = int(df.iloc[index, 7])
#         v9 = df.iloc[index, 8]
#         v10 = df.iloc[index, 9]
#         v11 = df.iloc[index, 10]
#         v12 = bool(df.iloc[index, 11])
#         v13 = int(df.iloc[index, 12])
#         v14 = int(df.iloc[index, 13])
#         v15 = int(df.iloc[index, 14])
#         v16 = df.iloc[index, 15]
#         v17 = df.iloc[index, 16]
#         v18 = df.iloc[index, 17]
#         v19 = source_table
#         sql = "INSERT INTO " + target_table + " (index, id, conversation_id, created_at, date, tweet, hashtags, user_id, username, name, link, retweet, nlikes, nreplies, nretweets, quote_url, user_rt_id, user_rt, from_staging_table) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
#         cursor.execute(sql, (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19))
#         connection.commit()
#     cursor.close()
#     finishtime = datetime.now()
#     # print ("Time Used: " + str(finishtime-start_time))
#     # if len(df) == len(df_source):
#     #     #print ("WARNING: Gap detected between lates FACT entry and newest STAGING entry")
#     #     warnings.warn("WARNING: Gap detected between lates FACT entry and newest STAGING entry")
#     # else:
#     #     print("No gap between staging and facts detected.")
#     print(str(len(df)) + " lines written to " + target_table + "; Number of lines in Staging Table: " + str(
#         len(df_source)))
#     return df_duplicates
#
#
# def insert_new_tweets_into2(table_name, connection, df_long, df_short):
#     '''
#     fügt neue Twitter Daten den alten hinzu, dropt die bisherige Tabelle und ersetzt sie durch die concatinierten Datamarts
#     Einfach zu implementieren aber langsam
#     '''
#     start_time = datetime.now()
#     df = df_long[df_long.id.isin(df_short.id) == False]
#     # df = df.iloc[:,0]
#     # df2 = df.iloc[:, 0:-1]
#     # df_new.iloc[:, 0]
#     result = pd.concat([df, df_short])
#     cursor = connection.cursor()
#     sql = "drop table " + table_name
#     print(sql)
#     cursor.execute(sql)
#     connection.commit()
#     result.to_sql(table_name, con=engine)  # writes df to Database using SQL alchemy engine
#     finishtime = datetime.now()
#     print("Time Used: " + str(finishtime - start_time))
#
#
# def update_process(source_table, target_table):
#     '''
#     loads latest staging table and adds new tweets to facts
#     :param source_table: Name der Staging Tabelle
#     :param target_table: Tabelle in die geschrieben wird. Default Wert 'facts'
#     :return:
#     '''
#     # hashtag = re.search(r"s_h_([a-z \d]+)_", source_table).group(1)  # extracts hashtag from source_table name
#
#     # Part1: Nimmt insert in Faktentablle vor. Schreibt die Duplikate zurück in df_duplictaes
#     engine, connection, metadata = init_db_connections()
#     df_duplicates = insert_new_tweets_into(source_table, connection,
#                                            target_table)  # nimmt die angegeben Staging Tabelle und überführt deren neue Werte nach Tabelle facts #Time Used: 0:00:00.013012
#
#     # Part2: Aktualisiert die bereits vorhandenen Einträge der Faktentabelle mit den neuen Werten (z.B. Likes) der Duplikate
#     drop_table("duplicates_temp", engine, metadata)
#     df_duplicates.to_sql("duplicates_temp", con=engine)
#     sql = "update " + target_table + """ set nlikes = duplicates_temp.nlikes, nreplies = duplicates_temp.nreplies, nretweets = duplicates_temp.nretweets, from_staging_table = duplicates_temp.staging_name from duplicates_temp where """ + target_table + ".id = duplicates_temp.id"
#     cursor = connection.cursor()
#     cursor.execute(sql)
#     connection.commit()
#     drop_table("duplicates_temp", engine, metadata)
#     cursor.close()
#     db_close(connection)
#     print(str(len(
#         df_duplicates)) + " rows were updated in " + target_table + " since " + source_table + " had more recent values for these.")
#
#     # Part3: Aktualisiert die overlap dtetection view. Sie zeigt, welche ID in welcher Staging tabelle ist
#     create_overlap_detection_view()  # refreshes the overlap view
#
#     # def explain_duplicates(source_table, df_duplicates):
#     #     df_duplicates = df_duplicates.reset_index(drop=True) #resets series index so it can be merged with another series
#     #     comma_list = ["," for item in range(len(df_duplicates))]  # creates a list that contains nothing but the word "union"
#     #     df = pd.concat([df_duplicates, pd.Series(comma_list)], axis=1).reset_index() #merges comma series with df_duplicates series
#     #     df.iloc[len(df) - 1, 2] = ""  # replaces the comma union with "". Otherwise the sql statement would end commma
#     #     hashtag = re.search(r"s_h_([a-z \d]+)_", source_table).group(1) #extracts hashtag from source_table name
#     #     id_list=""
#     #     for index, element in df.iterrows():
#     #         id_list += """'"""+element.iloc[1:2].values+"""'""" + element.iloc[2:3].values #concatenates all id for SQL IN statement
#     #     sql = """select * from v_hashtag_overlap_detection where id IN (""" + id_list + """) and staging_name like '%""" + hashtag + """%'"""
#     #     sql2 = str(sql[0:1]) #for some reason the SQL statement end up a numpy array. Here I'm correctiong that
#     #     connection = db_connect()
#     #     df_result  = sqlio.read_sql_query(sql, connection)
#     #     print ("bing")
#
#
# def Since_Date_Calculator(hashtag="a", username="a"):
#     # hashtag = "b2908"
#     hashtag = hashtag.lower()
#     username = username.lower()
#     if hashtag != "a":
#         # hastag mode
#         sql = """SELECT right(max(tablename),13) FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema' and tablename like 's_h_""" + hashtag + """%'"""
#     else:
#         # username mode
#         sql = """SELECT right(max(tablename),13) FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema' and tablename like 's_""" + username + """%'"""
#     print(sql)
#     connection = db_connect()
#     df_result = sqlio.read_sql_query(sql, connection)
#     sql = df_result.iloc[0, 0]
#     if sql == None:
#         date_object = 0  # no fact table for this hashtag so far
#     else:
#         date_object = datetime.strptime(sql, "%Y%m%d_%H%M")  # converts  date which Twint will use as Since value
#         date_object = date_object - timedelta(hours=1, minutes=0)  # adds atrificial overlap as safetymarging
#     db_close(connection)
#     return date_object
#
#
# def get_facts(username, year):
#     '''
#     Ruft definierte measures aus der Faktentabelle ab
#     :param username:
#     :param year:
#     :return: A: dataframe aller spalten, B:nlikes LIST, C:nreplies LIST, D:nretweets LIST
#     '''
#     engine, connection, metadata = init_db_connections()
#     # sql = '''SELECT * FROM public.facts where username = '%s';''' % username
#     sql = '''SELECT * FROM public.facts where username = '%s' and left(date,4) = '%s' ;''' % (username, year)
#     df = sqlio.read_sql_query(sql, connection)
#     nlikes = df.nlikes
#     nreplies = df.nreplies
#     nretweets = df.nretweets
#     return df, nlikes.tolist(), nreplies.tolist(), nretweets.tolist()
#
#
# def create_overlap_detection_view():
#     '''
#     creates a view in Postgres that contains all staging tables and checks, if the tweet is already known
#     '''
#     # Part 1: Retrieves list of staging tables that store hashtags
#     sql = """SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname != 'pg_catalog' AND schemaname != 'information_schema' and tablename like 's_h_%';"""
#     connection = db_connect()
#     cursor = connection.cursor()
#     cursor.execute(sql)
#     df = sqlio.read_sql_query(sql, connection)  # loads list of tables into DF
#     # Part 2: creates view in DB
#     union_list = ["union" for item in range(len(df))]  # creates a list that contains nothing but the word "union"
#     df = df.assign(union=pd.Series(union_list).values)
#     df.iloc[len(
#         df) - 1, 1] = ";"  # replaces the last union with ";". Otherwise the sql statement would end with union and hence not run
#     sql = "CREATE OR REPLACE VIEW v_hashtag_overlap_detection AS "
#     for index, element in df.iterrows():
#         sql += "select id, staging_name from " + element[0] + " " + element[1] + " "
#     cursor.execute(sql)
#     connection.commit()
#     cursor.close()
#     print("Overlap Detection View refreshed. See view v_hashtag_overlap_detection for details.")
#
#
# def delte_user_duplicates():
#     sql = "delete from n_users where id in (select id as min_last_seen from n_users group by id having count(id) > 1) and last_seen in (select min(last_seen) as min_last_seen from n_users group by id having count(id) > 1)"
#     update_table(sql)
#
#
# def purge_scores():
#     """
#     sets all scores to back to 0
#     :return:
#     """
#     update_table("update n_users set gen1 = 0, gen2 = 0, gen3 = 0, score_sum = 0, opinion_count = 0")  # score reset
#     # update_table("update n_users set gen2 = 0")  # score reset
#     # update_table("update n_users set gen3 = 0")  # score reset
#     # update_table("update n_users set score_sum = 0")  # score reset
#     # update_table("update n_users set opinion_count = 0")  # score reset
#     print("All scores purged.")
#
#
# def purge_aut_scores():
#     """
#     sets all scores to back to 0
#     :return:
#     """
#     update_table("update n_users set gen1_aut = 0, gen2_aut = 0, gen3_aut = 0, score_sum_aut = 0, opinion_count_aut = 0")  # score reset
#     # update_table("update n_users set gen1_aut = 0")  # score reset
#     # update_table("update n_users set gen2_aut = 0")  # score reset
#     # update_table("update n_users set gen3_aut = 0")  # score reset
#     # update_table("update n_users set score_sum_aut = 0")  # score reset
#     # update_table("update n_users set opinion_count_aut = 0")  # score reset
#     # print("All aut scores purged.")
#
#
# def score_change2(factor, friend_score, constant, core_score):
#     # constant = constant * np.sign((core_score - (friend_score)) / friend_score)
#     if friend_score == 0 and core_score >= 0:
#         friend_score=1
#     else:
#         friend_score = -1
#     try:
#         # new_score = (factor * friend_score + constant) * np.sign((core_score - (friend_score)) / friend_score)
#         new_score = (factor * friend_score + constant) / ((core_score - friend_score) / 10) * np.sign(
#             (core_score - (friend_score)) / friend_score)
#     except TypeError:
#         new_score = 0
#     # print("New Score Formula:" + str(new_score))
#     except ZeroDivisionError:
#         new_score = 0
#     if np.isnan(new_score):
#         return 0
#     return (new_score)
#
# def score_change3(factor, friend_score, constant, core_score, weight, gen):
#     """
#     This version is for generation 3 scores only
#     Since weight is a divisor, high weight means big score change (imagine the weight as the rope weight in a tug of war)
#     factor = manueller gewichtungs faktor
#     friend_score = score des freundes, von dem wir punkte haben wollen
#     constant = Zweck: Vermeiden von divsion duch null. Wert für gewöhnlich 1
#     weight = "Seilgewicht". Soll dazu dienen, dass weit entfernte user sich weniger stark beinflussen. Hauptsächlich, damit die extremen nicht alle aus der Mitte ziehen
#     """
#     # constant = constant * np.sign((core_score - (friend_score)) / friend_score)
#     if friend_score == 0 and core_score >= 0:
#         friend_score=1
#     if friend_score == 0 and core_score < 0:
#         friend_score = -1
#     if friend_score == core_score:
#         friend_score += 1
#     # if (core_score - friend_score) / weight) < 1 and (core_score - friend_score) / weight) > -1:
#     #     calculated_weight = 1
#
#     calculated_weight = (core_score - friend_score) / weight
#     #weight is rounded away from zero
#     rounded_weight  = abs(float(Decimal(calculated_weight).quantize(Decimal(0), rounding=ROUND_UP)))
#     try:
#         # new_score = (factor * friend_score + constant) * np.sign((core_score - (friend_score)) / friend_score)
#         #if gen == 3:
#         #new_score = (factor * friend_score + constant) / (rounded_weight) * np.sign(core_score * (friend_score))
#         new_score = core_score / 10
#
#
#         #bei dieser version bekommt der new_score das vorzeichen des friend_scores
#         #new_score = (factor * friend_score + constant) / (rounded_weight) * np.sign(friend_score)
#
#         #elif gen==2:
#         #    new_score = (factor * friend_score + constant) / ((core_score - friend_score) / weight) * np.sign((friend_score - (core_score)) / friend_score)
#     except TypeError:
#         new_score = 0
#     # print("New Score Formula:" + str(new_score))
#     except ZeroDivisionError:
#         new_score = 0
#     if np.isnan(new_score):
#         return 0
#     return (new_score)
#
#
# def score_change4(friend_score, core_score, number_of_common_friends, op_count = 0):
#
#     #friend_score = score des users, der bewertet wird
#     #core_score = score des users, der die bewertung abgibt
#     if friend_score == 0 and core_score >= 0:
#         friend_score=1
#     if friend_score == 0 and core_score < 0:
#         friend_score = -1
#     if friend_score == core_score:
#         friend_score += 1
#
#     #ermittelt prozentualen anteil gemeinsamer freunde zwischen friend (bekomt score) und core (gibt score)
#     #sql = "with user1 as (select follows_ids from n_friends where user_id = " + str(core_id) + "), user2 as (select follows_ids from n_friends where user_id = " + str(friend_id) + ")select cast(c.count as numeric) /  cast (a.count as numeric) as percent_match from (select count(*) from user1) as a, (select count (*) from (select * from user1 intersect select * from user2)b) c"
#     #df = select_from_db(sql)
#
#     #je mehr gemeinsame freunde core und friend haben, desto mehr score gibt der core an den friend
#     if int(number_of_common_friends) <= 10: #10
#         score_divisor = 5 #20
#     elif int(number_of_common_friends) <= 25: #20
#         score_divisor = 2 #10
#     elif int(number_of_common_friends) > 25: #30
#         score_divisor = 1 #2
#
#     # if int(number_of_common_friends) <= 10: #10
#     #     score_divisor = 10 #20
#     # elif int(number_of_common_friends) <= 15: #20
#     #     score_divisor = 5 #10
#     # elif int(number_of_common_friends) <= 25: #30
#     #     score_divisor = 2 #5
#     # elif int(number_of_common_friends) > 25: #30
#     #     score_divisor = 1 #2
#
#     # if df.values[0, 0] <= 0.1:
#     #     score_divisor = 20
#     # elif df.values[0, 0] <= 0.2:
#     #     score_divisor = 10
#     # elif df.values[0, 0] <= 0.3:
#     #     score_divisor = 5
#     # elif df.values[0, 0] > 0.3:
#     #     score_divisor = 2
#
#     if op_count > 0:
#         core_score = core_score / op_count
#     try:
#         new_score = core_score / score_divisor
#     except TypeError:
#         new_score = 0
#     # print("New Score Formula:" + str(new_score))
#     except ZeroDivisionError:
#         new_score = 0
#     if np.isnan(new_score):
#         return 0
#     return (new_score)
#
#
# def set_gen_one_scores(gen):
#     """
#     :param gen: either 1 or 2 for first or second generation of scores
#     :return:
#     """
#     timestamp1 = datetime.now().strftime("%Y%m%d_%H%M%S")
#     delte_user_duplicates()
#     if gen == 1:
#         update_table("update n_users set gen1 = 0")  # score reset
#         update_table("update n_users set gen1 = n_cores.score, score_sum = n_cores.score from n_cores where id = n_cores.user_id")
#         update_table("update n_users set opinion_count = 0")  # score reset
#
#         #sql = "select distinct user_id, score, 0 as opinion_count from n_cores where admin_comment like '%Union%'"
#         sql = "select distinct user_id, score, 0 as opinion_count from n_cores"
#         df = select_from_db(sql)  # list of users who will pass on part of their score
#         sql_friends_of_all_cores = "select distinct u.id, u.gen1, 0 as opinion_count  from n_cores c, n_friends f, n_users u where cast (f.follows_ids as bigint) =  u.id and c.user_id = f.user_id"
#     elif gen == 2:
#         update_table("update n_users set gen2 = 0")  # score reset
#         #df = select_from_db("select distinct u1.id, gen1, opinion_count from n_users u1")
#         df = select_from_db("select distinct u1.id, gen1, opinion_count from n_users u1 where id in (select distinct user_id from facts_hashtags where from_staging_table like '%sturm%')")
#
#         #sql = "select distinct u1.id, gen1, opinion_count from n_users u1 where id in (select distinct user_id from facts_hashtags where from_staging_table like '%sturm%' and user_id = 19108766) "
#         #df = select_from_db(sql)
#
#         # df = select_from_db("select distinct u1.id, gen1, 0 as opinion_count from n_users u1 where  (u1.gen1 > 5 or  u1.gen1 < -5)")  # list of users who will pass on part of their score
#         # df = select_from_db("select distinct u1.id, gen1, 0 as opinion_count from n_users u1 where  (u1.score_sum > 5 or  u1.score_sum < -5)") #3rd generation
#         # df = select_from_db("select distinct u1.id, gen1 from n_users u1 where  (u1.gen1 > 5 or  u1.gen1 < -5) and id = 1107268454")
#         #sql_friends_of_all_cores = "select distinct cast(follows_ids as bigint) as id, 0 as gen1, 0 as opinion_count from n_friends where user_id in (select id from n_users u1 where (u1.gen1 > 5 or  u1.gen1 < -5))"
#         sql_friends_of_all_cores = "select distinct cast(follows_ids as bigint) as id, 0 as gen1, 0 as opinion_count from n_friends where user_id in (select id from n_users u1 where (u1.opinion_count >= 5 or  u1.opinion_count < -5))"
#         # sql_friends_of_all_cores = "select distinct cast(follows_ids as bigint) as id, 0 as gen1 from n_friends where user_id in (select id from n_users u1 where (u1.gen1 > 5 or  u1.gen1 < -5) and id = 1107268454)" #limitierte version des obigen statements
#
#     df_update_this = select_from_db(sql_friends_of_all_cores)
#     dict_update_this = df_update_this.set_index('id').to_dict()
#     count = 0
#
#     pbar = tqdm(total=len(df))
#
#     for index, element in df.iterrows(): #contains a core
#         count += 1
#
#         if gen ==1 or gen ==2:
#             sql_friens_of_one_core = "select cast(u.id as bigint) as id, cast (u.score_sum as bigint) as score_sum from n_friends f, n_users u where cast (f.follows_ids as bigint) = u.id and user_id = " + str(element[0])
#         if gen == 3:
#             sql_friens_of_one_core = "select cast(u.id as bigint) as id, cast (u.score_sum as bigint) as score_sum from n_friends f, n_users u where cast (f.follows_ids as bigint) = u.id and user_id = " + str(element[0]) + " and (u.opinion_count <= 5)"
#         df_score_change = select_from_db(sql_friens_of_one_core)
#         for index2, element2 in df_score_change.iterrows(): #contains friends of a core
#
#             if gen == 1:
#                 factor = 0.3
#             else:
#                 if element[2] >= 25:
#                     factor = 0.2
#                 elif element[2] >=5:
#                     factor = 0.1
#                 else:
#                     factor = 0
#
#             score_change = score_change2(factor=factor, friend_score=element2[1], constant=1, core_score=element[1])
#
#             try:
#                 #dict_update_this['gen1'][17965092]
#                 # if int(element2[0]) == 143000381:
#                 #     print ("TEST")
#                 #     print ("STOP 143000381")
#                 dict_update_this['gen1'][int(element2[0])] = dict_update_this['gen1'][int(element2[0])] + float(score_change)
#                 dict_update_this['opinion_count'][int(element2[0])] = dict_update_this['opinion_count'][int(element2[0])] + 1  # counts how many scores a user has gotten
#             except KeyError:
#                 print("Probably we don't know any friends of user " + str(int(element2[0])))
#         pbar.update(1)
#
#     df = pd.DataFrame.from_dict(dict_update_this)
#     engine, metadata = sql_alchemy_engine()
#     drop_table('temp_scores', engine, metadata)
#     df.to_sql('temp_scores', con=engine)  # writes df to Database using SQL alchemy engine
#
#     update_table("update n_users set gen" + str(
#         gen) + " = temp_scores.gen1, opinion_count = n_users.opinion_count + temp_scores.opinion_count from temp_scores where n_users.id = temp_scores.index")
#     update_table("update n_users set score_sum = gen1 + gen2 + gen3")
#     drop_table('temp_scores', engine, metadata)
#
#     print("Process Start: " + timestamp1)
#     timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
#     print("Process End: " + timestamp2)
#     # duration =  timestamp2 - timestamp1
#     # print (duration.strftime("%Y%m%d"))
#
# def set_gen_three_scores(gen, df_common_friends = 0, aut = False):
#     """
#     wird für alle generation benutzt, auch wenn der Name etwas anderes sagt
#     :param gen: either 1 or 2 or 3  for  generation of scores
#     :param df_common_friends: Sher große tabelle mit anzahl gemeinsamer freunden, wo diese bekannt sind
#     :param aut: True: Autoritäts Score berechnen; False: Links Rechts Score berechnen
#     :return:
#     """
#     timestamp1 = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     # can be used for aut and right/left score calculation
#
#     if aut == True:
#     #aut scores statements
#         if gen == 1:
#             #update_table("update n_users set gen1_aut = n_cores.aut_score, score_sum_aut = n_cores.aut_score from n_cores where id = n_cores.user_id")
#             #sql = "select f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum_aut as score_sum_friend, u.opinion_count_aut as opinion_count_friend from n_cores c, n_friends f, n_users u where c.user_id = f.user_id and cast (f.follows_ids as bigint) = u.id"
#             sql = "select f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum_aut as score_sum_friend, u.opinion_count_aut as opinion_count_friend from n_cores c, n_friends f, n_users u where c.user_id = f.user_id and cast (f.follows_ids as bigint) = u.id"
#         elif gen == 2:
#             #update_table("update n_users set gen2_aut = 0")
#             # alle cores (die opinion_count >= 5 haben) mit ihren scores und allen usern, den sie folgen
#             sql = "select follows_ids as hashtag_users, n_friends.user_id as core, n_users.score_sum_aut, n_users.opinion_count_aut, n_users2.score_sum_aut as score_sum_friend , n_users2.opinion_count_aut as opinion_count_friend from n_friends, n_users, n_users as n_users2 where n_users.id = n_friends.user_id  and n_users.opinion_count_aut >= 5 and cast (n_friends.follows_ids as bigint) = n_users2.id"
#         elif gen == 3:
#             #update_table("update n_users set gen3_aut = 0")
#             # All users with a low opinion count for one hashtag with people who they follow that have an opinion count >= 5
#             sql = "select follows_ids as hashtag_users, user_id as core, n_users.score_sum_aut, n_users.opinion_count_aut from n_friends, n_users where cast (follows_ids as bigint) in (select f.user_id as id from facts_hashtags f , n_users u where f.from_staging_table like '%sturm%' and f.user_id = u.id except select id from n_users where opinion_count_aut >= 5) and n_users.id = n_friends.user_id and n_users.opinion_count_aut >=5 order by hashtag_users"
#     else:
#     # L/R scores statements
#         if gen == 1:
#             #update_table("update n_users set gen1 = n_cores.score, score_sum = n_cores.score from n_cores where id = n_cores.user_id")
#             #sql2 = "select admin_comment, f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum as score_sum_friend, u.opinion_count as opinion_count_friend from n_cores c, n_friends f, n_users u where c.user_id = f.user_id and cast (f.follows_ids as bigint) = u.id"
#             #df2 = select_from_db(sql2)
#
#             sql = "select f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum as score_sum_friend, u.opinion_count as opinion_count_friend from n_cores c, n_friends f, n_users u where c.user_id = f.user_id and cast (f.follows_ids as bigint) = u.id"
#             #sql = "select f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum as score_sum_friend, 	u.opinion_count as opinion_count_friend from n_cores c, n_friends f, n_users u where c.user_id = f.user_id and 	cast (f.follows_ids as bigint) = u.id and c.user_id = 136266976 and f.follows_ids = '4617361'"
#         elif gen == 2:
#             #update_table("update n_users set gen2 = 0")
#             #alle cores (die opinion_count >= 5 haben) mit ihren scores und allen usern, den sie folgen
#             sql = "select follows_ids as hashtag_users, n_friends.user_id as core, n_users.score_sum, n_users.opinion_count, n_users2.score_sum as score_sum_friend , n_users2.opinion_count as opinion_count_friend from n_friends, n_users, n_users as n_users2 where n_users.id = n_friends.user_id  and n_users.opinion_count >= 5 and cast (n_friends.follows_ids as bigint) = n_users2.id"
#             #sql = "select follows_ids as hashtag_users, n_friends.user_id as core, n_users.score_sum, n_users.opinion_count, n_users2.score_sum as score_sum_friend , n_users2.opinion_count as opinion_count_friend from n_friends, n_users, n_users as n_users2 where n_users.id = n_friends.user_id  and n_users.opinion_count >= 5 and cast (n_friends.follows_ids as bigint) = n_users2.id and n_friends.user_id = 2281250150 and n_users2.id = 778264709443252224"
#         elif gen == 3:
#             #WIRD NICHT MEHR BENUTZT! STATTDESSEN GEN 4 NUTZEN
#             #update_table("update n_users set gen3 = 0")
#             # All users with a low opinion count for one hashtag with people who they follow that have an opinion count >= 5
#             sql = "select follows_ids as hashtag_users, user_id as core, n_users.score_sum, n_users.opinion_count from n_friends, n_users where cast (follows_ids as bigint) in (select f.user_id as id from facts_hashtags f , n_users u where f.from_staging_table like '%sturm%' and f.user_id = u.id except select id from n_users where opinion_count >= 5) and n_users.id = n_friends.user_id and n_users.opinion_count >=5 order by hashtag_users"
#         elif gen == 4:
#             #Punkte werden auf Basis dessen vergeben, wem ich folge. Punkte werden nur an user vergeben, von denen wir <= 50 opinions haben
#             sql = "select f.follows_ids as hashtag_users, f.user_id as core, u.score_sum, u.opinion_count, u2.score_sum as score_sum_friend , u2.opinion_count as opinion_count_friend from n_users u, n_followers f, n_users u2 where cast (f.follows_ids as bigint) = u.id and f.user_id = u2.id and u2.opinion_count >= 5 and u.opinion_count <= 200"
#     df = select_from_db(sql)
#     pbar = tqdm(total=len(df))
#     dict_update_this = {id: {"gen1": 0, "opinion_count": 0} for id in df.hashtag_users.to_list()}
#
#     if isinstance(df_common_friends, int):
#         pass
#     else:
#         cores = df_common_friends.core_id.to_list()
#         friends = df_common_friends.friend_id.to_list()
#         friend_number = df_common_friends.common_friends.to_list()
#         coresNfriends = []
#         #cores und friends in einer list verbinden
#         for index, element in enumerate (cores):
#             cores_sub = str(element)
#             friends_sub = str(friends[index])
#             coresNfriends.append(cores_sub+friends_sub)
#         dict_common_friends = dict(zip(coresNfriends, friend_number))
#
#
#     # dict_common_friends = {}
#     #
#     # dict_sample = {1: 'mango', 2: 'pawpaw'}
#     #
#     # >> > keys = ['a', 'b', 'c']
#     # >> > values = [1, 2, 3]
#     # >> > dictionary = dict(zip(keys, values))
#     # >> > print(dictionary)
#     # {'a': 1, 'b': 2, 'c': 3}
#
#
#     for index, element in df.iterrows(): #contains a core
#         # if element[0] == '85654678':
#         # if element[0] ==  '944181149424898048':
#         #     print ("Stopp")
#         #     print("Stopp")
#         if element[3] >= 25:
#             factor = 1 #0.5
#         elif element[3] >= 5:
#             factor = 0.25 #0.25
#         else:
#             if gen == 1:
#                 factor = 1 #0.5
#             else:
#                 factor = 0
#
#         if gen == 2 or gen == 1:
#             #def score_change4(friend_score, core_score, friend_id, core_id):
#             #number_of_common_friends = df_common_friends[df_common_friends['core'] == element[1]]
#             # if element[1] == 136266976:
#             #     print ("ALARRRRM!")
#             #     print("ALARRRRM!")
#             #     if element[0] == '4617361':
#             #        print("ALARRRRM!2")
#             #        print("ALARRRRM!2")
#
#             #df[df['core'] == element[1]]
#             #df[df['hashtag_users'] == 7226492]
#
#             if isinstance(df_common_friends, int):
#                 pass
#             else:
#                 with warnings.catch_warnings():
#                     warnings.simplefilter(action='ignore', category=FutureWarning)
#                     try:
#                         #number_of_common_friends = df_common_friends[(df_common_friends['core'] == element[1]) & (df_common_friends['friend'] == int(element[0]))].iloc[0,2]
#                         number_of_common_friends = dict_common_friends.get(str(element[1]) + element[0])
#                         if number_of_common_friends is None:
#                             number_of_common_friends = 0
#                     except:
#                         number_of_common_friends = 0
#
#             # if number_of_common_friends > 0:
#             #     xs_test = dict_common_friends.get(str(element[1]) + element[0])
#             #     print("TREFFER")
#
#             new_score = score_change4(friend_score=element[4], core_score=element[2], number_of_common_friends=number_of_common_friends)
#
#             #new_score = score_change3(factor=factor, friend_score=element[4], constant=1, core_score=element[2], weight=4, gen = 2)
#             #print (new_score)
#             # new_score = score_change2(factor=factor, friend_score=element[4], constant=1, core_score=element[2])
#             # print (new_score)
#         if gen == 3:
#             new_score = score_change3(factor=factor, friend_score = element[2], constant=1, core_score=0, weight = 4, gen=3)
#             #print (new_score)
#         if gen == 4:
#             #Berücksichtig den Opinion Count des Cores. Maximal Score, den der Core weiter geben kann, ist Score_Sum / Opinion Count
#             new_score = score_change4(friend_score=element[2], core_score=element[4], number_of_common_friends=0, op_count=element[5])
#         # new_score = score_change2(factor=factor, friend_score=element[2], constant=1, core_score=0)
#         dict_update_this[element[0]]['gen1'] = dict_update_this[element[0]]['gen1'] + float(new_score)
#         dict_update_this[element[0]]['opinion_count'] = dict_update_this[element[0]]['opinion_count'] + 1
#         new_score = 0
#         pbar.update(1)
#
#     df = pd.DataFrame.from_dict(dict_update_this)
#     df = df.transpose()
#     engine, metadata = sql_alchemy_engine()
#     drop_table('temp_scores', engine, metadata)
#     df.to_sql('temp_scores', con=engine)  # writes df to Database using SQL alchemy engine
#
#     #can be used for aut and right/left score calculation
#     if gen == 4:
#         #es gibt keine Spalte gen4. Daher nutzen wir hier die gen 3 spalte mit
#         gen = 3
#     if aut == True:
#         update_table("update n_users set gen" + str(gen) + "_aut = temp_scores.gen1, opinion_count_aut = n_users.opinion_count_aut + temp_scores.opinion_count from temp_scores where n_users.id = cast(temp_scores.index as bigint)")
#         update_table("update n_users set score_sum_aut = gen1_aut + gen2_aut + gen3_aut")
#     else:
#         update_table("update n_users set gen" + str(gen) + " = temp_scores.gen1, opinion_count = n_users.opinion_count + temp_scores.opinion_count from temp_scores where n_users.id = cast(temp_scores.index as bigint)")
#         update_table("update n_users set score_sum = gen1 + gen2 + gen3")
#     drop_table('temp_scores', engine, metadata)
#     print("Process Start: " + timestamp1)
#     timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
#     print("Process End: " + timestamp2)
#
# def set_plus_minus_scores(gen, aut = False):
#     """
#     wird für alle generation benutzt, auch wenn der Name etwas anderes sagt
#     :param gen: either 1 or 2 or 3  for  generation of scores
#     :param df_common_friends: Sher große tabelle mit anzahl gemeinsamer freunden, wo diese bekannt sind
#     :param aut: True: Autoritäts Score berechnen; False: Links Rechts Score berechnen
#     :return:
#     """
#     timestamp1 = datetime.now().strftime("%Y%m%d_%H%M%S")
#
#     # can be used for aut and right/left score calculation
#     if aut == True:
#     #aut scores statements
#         if gen == 1:
#             #update_table("update n_users set gen1_aut = n_cores.aut_score, score_sum_aut = n_cores_plus_minus.aut_score from n_cores_plus_minus where id = n_cores_plus_minus.user_id")
#             #sql = "select f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum_aut as score_sum_friend, u.opinion_count_aut as opinion_count_friend from n_cores_plus_minus c, n_friends f, n_users u where c.user_id = f.user_id and cast (f.follows_ids as bigint) = u.id"
#             sql = "select f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum_aut as score_sum_friend, u.opinion_count_aut as opinion_count_friend from n_cores_plus_minus c, n_friends f, n_users u where c.user_id = f.user_id and cast (f.follows_ids as bigint) = u.id"
#         elif gen == 2:
#             #update_table("update n_users set gen2_aut = 0")
#             # alle cores (die opinion_count >= 5 haben) mit ihren scores und allen usern, den sie folgen
#             sql = "select follows_ids as hashtag_users, n_friends.user_id as core, n_users.score_sum_aut, n_users.opinion_count_aut, n_users2.score_sum_aut as score_sum_friend , n_users2.opinion_count_aut as opinion_count_friend from n_friends, n_users, n_users as n_users2 where n_users.id = n_friends.user_id  and n_users.opinion_count_aut >= 5 and cast (n_friends.follows_ids as bigint) = n_users2.id"
#         elif gen == 3:
#             #update_table("update n_users set gen3_aut = 0")
#             # All users with a low opinion count for one hashtag with people who they follow that have an opinion count >= 5
#             sql = "select follows_ids as hashtag_users, user_id as core, n_users.score_sum_aut, n_users.opinion_count_aut from n_friends, n_users where cast (follows_ids as bigint) in (select f.user_id as id from facts_hashtags f , n_users u where f.from_staging_table like '%sturm%' and f.user_id = u.id except select id from n_users where opinion_count_aut >= 5) and n_users.id = n_friends.user_id and n_users.opinion_count_aut >=5 order by hashtag_users"
#     else:
#     # L/R scores statements
#         if gen == 1:
#             #update_table("update n_users set gen1 = n_cores_plus_minus.score, score_sum = n_cores_plus_minus.score from n_cores_plus_minus where id = n_cores_plus_minus.user_id")
#             #sql2 = "select admin_comment, f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum as score_sum_friend, u.opinion_count as opinion_count_friend from n_cores_plus_minus c, n_friends f, n_users u where c.user_id = f.user_id and cast (f.follows_ids as bigint) = u.id"
#             #df2 = select_from_db(sql2)
#
#             sql = "select f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum as score_sum_friend, u.opinion_count as opinion_count_friend from n_cores_plus_minus c, n_friends f, n_users u where c.user_id = f.user_id and cast (f.follows_ids as bigint) = u.id"
#             #sql = "select f.follows_ids as hashtag_users, c.user_id as core, c.score as score_sum, 0 as opinion_count, u.score_sum as score_sum_friend, 	u.opinion_count as opinion_count_friend from n_cores_plus_minus c, n_friends f, n_users u where c.user_id = f.user_id and 	cast (f.follows_ids as bigint) = u.id and c.user_id = 136266976 and f.follows_ids = '4617361'"
#         elif gen == 2:
#             #update_table("update n_users set gen2 = 0")
#             #alle cores (die opinion_count >= 5 haben) mit ihren scores und allen usern, den sie folgen
#             sql = "select follows_ids as hashtag_users, n_friends.user_id as core, n_users.score_sum, n_users.opinion_count, n_users2.score_sum as score_sum_friend , n_users2.opinion_count as opinion_count_friend from n_friends, n_users, n_users as n_users2 where n_users.id = n_friends.user_id  and n_users.opinion_count >= 5 and cast (n_friends.follows_ids as bigint) = n_users2.id"
#             #sql = "select follows_ids as hashtag_users, n_friends.user_id as core, n_users.score_sum, n_users.opinion_count, n_users2.score_sum as score_sum_friend , n_users2.opinion_count as opinion_count_friend from n_friends, n_users, n_users as n_users2 where n_users.id = n_friends.user_id  and n_users.opinion_count >= 5 and cast (n_friends.follows_ids as bigint) = n_users2.id and n_friends.user_id = 2281250150 and n_users2.id = 778264709443252224"
#         elif gen == 3:
#             #WIRD NICHT MEHR BENUTZT! STATTDESSEN GEN 4 NUTZEN
#             #update_table("update n_users set gen3 = 0")
#             # All users with a low opinion count for one hashtag with people who they follow that have an opinion count >= 5
#             sql = "select follows_ids as hashtag_users, user_id as core, n_users.score_sum, n_users.opinion_count from n_friends, n_users where cast (follows_ids as bigint) in (select f.user_id as id from facts_hashtags f , n_users u where f.from_staging_table like '%sturm%' and f.user_id = u.id except select id from n_users where opinion_count >= 5) and n_users.id = n_friends.user_id and n_users.opinion_count >=5 order by hashtag_users"
#         elif gen == 4:
#             #Punkte werden auf Basis dessen vergeben, wem ich folge. Punkte werden nur an user vergeben, von denen wir <= 50 opinions haben
#             sql = "select f.follows_ids as hashtag_users, f.user_id as core, u.score_sum, u.opinion_count, u2.score_sum as score_sum_friend , u2.opinion_count as opinion_count_friend from n_users u, n_followers f, n_users u2 where cast (f.follows_ids as bigint) = u.id and f.user_id = u2.id and u2.opinion_count >= 5 and u.opinion_count <= 200"
#     df = select_from_db(sql)
#     pbar = tqdm(total=len(df))
#     dict_update_this = {id: {"gen1": 0, "opinion_count": 0} for id in df.hashtag_users.to_list()}
#
#     for index, element in df.iterrows(): #contains a core
#         if element[3] >= 25:
#             factor = 1 #0.5
#         elif element[3] >= 5:
#             factor = 0.25 #0.25
#         else:
#             if gen == 1:
#                 factor = 1 #0.5
#             else:
#                 factor = 0
#
#         if gen == 2 or gen == 1:
#             #new_score = score_change4(friend_score=element[4], core_score=element[2], number_of_common_friends=number_of_common_friends)
#             new_score = score_change3(factor=factor, friend_score=element[4], constant=1, core_score=element[2], weight=4, gen = 2)
#             #print (new_score)
#             # new_score = score_change2(factor=factor, friend_score=element[4], constant=1, core_score=element[2])
#             # print (new_score)
#         if gen == 3:
#             new_score = score_change3(factor=factor, friend_score = element[2], constant=1, core_score=0, weight = 4, gen=3)
#             #print (new_score)
#         if gen == 4:
#             new_score = score_change3(factor=factor, friend_score = element[2], constant=1, core_score=0, weight = 4, gen=3)
#             #Berücksichtig den Opinion Count des Cores. Maximal Score, den der Core weiter geben kann, ist Score_Sum / Opinion Count
#             #new_score = score_change4(friend_score=element[2], core_score=element[4], number_of_common_friends=0, op_count=element[5])
#         # new_score = score_change2(factor=factor, friend_score=element[2], constant=1, core_score=0)
#         dict_update_this[element[0]]['gen1'] = dict_update_this[element[0]]['gen1'] + float(new_score)
#         dict_update_this[element[0]]['opinion_count'] = dict_update_this[element[0]]['opinion_count'] + 1
#         new_score = 0
#         pbar.update(1)
#
#     df = pd.DataFrame.from_dict(dict_update_this)
#     df = df.transpose()
#     engine, metadata = sql_alchemy_engine()
#     drop_table('temp_scores', engine, metadata)
#     df.to_sql('temp_scores', con=engine)  # writes df to Database using SQL alchemy engine
#
#     #can be used for aut and right/left score calculation
#     if gen == 4:
#         #es gibt keine Spalte gen4. Daher nutzen wir hier die gen 3 spalte mit
#         gen = 3
#     if aut == True:
#         update_table("update n_users set gen" + str(gen) + "_aut = temp_scores.gen1, opinion_count_aut = n_users.opinion_count_aut + temp_scores.opinion_count from temp_scores where n_users.id = cast(temp_scores.index as bigint)")
#         update_table("update n_users set score_sum_aut = gen1_aut + gen2_aut + gen3_aut")
#     else:
#         update_table("update n_users set gen" + str(gen) + " = temp_scores.gen1, opinion_count = n_users.opinion_count + temp_scores.opinion_count from temp_scores where n_users.id = cast(temp_scores.index as bigint)")
#         update_table("update n_users set score_sum = gen1 + gen2 + gen3")
#     drop_table('temp_scores', engine, metadata)
#     print("Process Start: " + timestamp1)
#     timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S")
#     print("Process End: " + timestamp2)
#
#
# def common_friends():
#     """
#     erstellt eine Tabelle mit Anzahl der gemeinsamen Freunde. Dauer bei 8000 distinct friend IDs 5 Stunden. Länge 78Mio(!) einträge
#     :return:
#     """
#     target_table = "h_common_friends"
#     sql_all_ids = "select distinct user_id from n_friends" # where common_friends is null limit 100"
#     df_all_ids = select_from_db(sql_all_ids)
#
#     #connection = db_connect()
#     engine, metadata = sql_alchemy_engine()
#     #cursor = connection.cursor()
#     #for index in tqdm (range (len(df_all_ids))):
#     for index, element in tqdm(df_all_ids.iterrows()):
#         #id =  df_all_ids.iloc[0, 0]
#         id =  element[0]
#         sql_friend_count = "with users1 as (select user_id, follows_ids from n_friends f where f.user_id = "+str(id)+"), users2 as (select user_id, follows_ids from n_friends f) select cast (users1.user_id as text) ||  cast (users2.user_id as text) as id, users1.user_id as core_id, users2.user_id as friend_id, count(users1.follows_ids) as common_friends from users1, users2 where users1.follows_ids = users2.follows_ids and users1.user_id <> users2.user_id group by users1.user_id, users2.user_id"
#         df_all_friend_counts = select_from_db(sql_friend_count)
#         df_all_friend_counts.to_sql(target_table, engine, index = False, if_exists='append')
#
#     #todo: fertige in n_friends mit update markieren
#     #wo speichere ich das denn? =>     #h_common friends
#
# def get_user_id_and_date_for_tweet(table_name):
#     """
#     Nach function hashtag_absaugen() benutzten, um automatisch user_id und tweet date zu den heruntergeladenen tweets zu ergänzen
#     :param table_name: Name der Tabelle, in der Tweets und Datümer ergänzt werden sollen
#     :return:
#     """
#     sql = "select distinct  id, user_id, date from " + table_name + " where user_id = ''"
#     df = select_from_db(sql)
#     for i, element in df.iterrows():
#         user_id, t_date = TwitterAPI.API_get_user_id_via_tweet_id(element[0])
#         df.iloc[i:i + 1, 1:2] = user_id
#         df.iloc[i:i + 1, 2:3] = t_date
#     engine, metadata = sql_alchemy_engine()
#     drop_table('temp_ids', engine, metadata)
#     df.to_sql('temp_ids', con=engine)  # writes df to Database using SQL alchemy engine
#     update_table("update " + table_name + " t set user_id = a.user_id, date = cast (a.date as text) from temp_ids a where t.id = a.id")
#     drop_table('temp_ids', engine, metadata)
#
#
#
#
# def save_pickle(df, filename):
#     """
#     :param filename: filename to load
#     :return: none
#     """
#     with open(filename, 'wb') as file:
#         pickle.dump(df, file)
#
# def load_pickle(filename):
#     """
#     :param filename: filename to laod
#     :return: loaded file
#     """
#     with open(filename, 'rb') as file:
#         loaded_file = pickle.load(file)
#     return loaded_file
#



if __name__ == '__main__':
    pass
