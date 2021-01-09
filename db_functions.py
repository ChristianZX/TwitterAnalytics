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

def save_pickle(df, filename):
    """
    :param filename: filename to load
    :return: none
    """
    with open(filename, 'wb') as file:
        pickle.dump(df, file)


def load_pickle(filename):
    """
    :param filename: filename to laod
    :return: loaded file
    """
    with open(filename, 'rb') as file:
        loaded_file = pickle.load(file)
    return loaded_file