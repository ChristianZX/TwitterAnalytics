from datetime import datetime, timedelta
import pandas as pd
import pandas.io.sql as sqlio
import psycopg2
import math
from sqlalchemy import create_engine, Table, MetaData
import sys
import pickle
import API_KEYS


def df_to_sql(df: pd.DataFrame, tablename: str, drop="none", bulk_size: int = 1000000):
    """
    Writes Dataframe to DB
    :param df: name of dataframe that will be written to DB
    :param tablename: tablename for DB
    :param drop: 'replace' , 'append' or False to raise error when table already exists
    """
    if len (df) == 0:
        print ("Error: Can't write to DB since file size is 0.")
        sys.exit()
    engine, metadata = sql_alchemy_engine()
    if drop == "replace":
        drop_table(tablename)
        df.to_sql(tablename, con=engine)
    elif drop == "none":
        df.to_sql(tablename, con=engine)
    elif drop == "append":
        df.to_sql(tablename, con=engine, index=False, if_exists='append', chunksize=bulk_size)
    else:
        print("Error raised by df_to_sql(): None of the write conditions (replace, drop, append) was met. "
              "Check function parameters!")
        engine.dispose()
        sys.exit()
    engine.dispose()


def init_db_connections() -> tuple:
    """
    Convenience function so I dont have to call sql_alchemy_engine() and db_connect() separately
    :return: engine, connection, metadata
    """
    engine, metadata = sql_alchemy_engine()
    connection = db_connect()
    return engine, connection, metadata


def sql_alchemy_engine() -> tuple:
    """
    Replace engine_string with code following the example below
    engine = create_engine('postgresql://postgres:YOUR_PASSWORD@localhost:5433/tw_lytics01')
    :return: engine, metadata
    """
    engine = create_engine(API_KEYS.engine_string)
    metadata = MetaData(engine)
    return engine, metadata


def staging_timestamp() -> datetime.timestamp:
    """
    Creates Timestamp in format "%Y%m%d_%H%M"
    :return: timestamp
    """
    my_timestamp = datetime.now()
    my_timestamp = my_timestamp.strftime("%Y%m%d_%H%M")
    return my_timestamp


def db_connect():
    """
    Replace connection_string with code following below example:
        connection = psycopg2.connect(user="postgres",
                                  password="YOUR_PASSWORD",
                                  host="127.0.0.1",
                                  port="5433",
                                  database="YOUR_DB_NAME")
    Hint: often it's port 5432
    :return: psycopg2 connection
    """
    try:
        connection = psycopg2.connect(user="postgres",
                                             password=API_KEYS.db_password,
                                             host="127.0.0.1",
                                             port="5433",
                                             database="tw_lytics01")
        #connection = API_KEYS.connection_string
        cursor = connection.cursor()
        cursor.execute("SELECT version();")

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", str(error))
    return connection


def db_close(connection):
    try:
        # closing database connection.
        if connection:
            connection.close()
            print("PostgreSQL connection is closed")
    except:
        pass
        # print ("It seems no DB connection not was open.")


def drop_table(name):
    """
    :param name: Table name to be dropped
    :param engine: optional DB engine (to be removed in next version)
    :param metadata: optional DB metadata (to be removed in next version)
    """
    engine, metadata = sql_alchemy_engine()
    table_to_drop = Table(name, metadata)
    table_to_drop.drop(engine, checkfirst=True)


def select_from_db(sql) -> pd.DataFrame:
    """
    Sends an SQL Statement to the DB
    :param sql: SQL Statement that will be executed by the DB
    :return: Dataframe with Query Result
    """
    connection = db_connect()
    df = sqlio.read_sql_query(sql, connection)  # staging table
    db_close(connection)
    return df


def update_table(sql):
    """
    Send Update Statement to DB. Can also be used to send inserts, create tables and so on.
    :param sql: sql statement to be send to DB
    :return: none
    """
    connection = db_connect()
    cursor = connection.cursor()
    print(f"Update starts now with statement: {sql}")
    cursor.execute(sql)
    connection.commit()
    cursor.close()
    db_close(connection)
    print("Update completed.")


def update_to_invalid(cur_date, user_id):
    """
    Used to set columns to invalid in table n_users. These columns are not used anymore during next run.
    Used for non-German tweets and tweets of deleted users
    :return:
    """
    if math.isnan(user_id) == False:
        sql_invalid = f"update n_users set lr = 'invalid', pol = 'invalid', lr_pol_last_analysed = '{cur_date}' " \
                      f"where id  = {user_id}"
        update_table(sql_invalid)
        print(f"Set {user_id} to invalid")


def get_staging_table_name(table_name: str) -> str:
    """
    Adds prefix and suffix to table name. Example for hashtag le0711: s_h_le0711_20201128_2219
    :param table_name: central name part of staging table
    :return: table name with prefix and suffix
    """
    return 's_h_' + table_name + '_' + str(staging_timestamp())


def create_empty_staging_table(table_name: str) -> None:
    """
    Creates empty table with staging table layout.
    :param table_name: table name to be used in DB
    :return: none
    """
    sql = f'CREATE TABLE {table_name} (   index bigint,    id bigint,    conversation_id bigint,    created_at text ' \
          'COLLATE pg_catalog."default",    date text COLLATE pg_catalog."default",     tweet text COLLATE ' \
          'pg_catalog."default",    hashtags text COLLATE pg_catalog."default",    user_id bigint,    username text ' \
          'COLLATE pg_catalog."default",    name text COLLATE pg_catalog."default",    link bigint,    ' \
          'retweet bigint,    nlikes bigint,    nreplies bigint,    nretweets bigint,    quote_url bigint,    ' \
          'user_rt_id bigint,    user_rt bigint,    staging_name text COLLATE pg_catalog."default")'
    update_table(sql)


def save_pickle(df: pd.DataFrame, filename: str) -> None:
    """
    :param filename: filename to load
    :return: none
    """
    with open(filename, 'wb') as file:
        pickle.dump(df, file)


def load_pickle(filename: str):
    """
    :param filename: filename to laod
    :return: loaded file
    """
    with open(filename, 'rb') as file:
        loaded_file = pickle.load(file)
    return loaded_file


def insert_to_table_followers_or_friends(df, table_name, username):
    """
    Inserts new Followers into table n_followers
    :param df: dataframe with new followers
    :param table_name: Name of table to insert into. Structure works for n_followers and n_friends.
    :param username: True if username is known, False if not
    :return:
    """
    connection = db_connect()
    cursor = connection.cursor()

    if username is False:
        sql = f"INSERT INTO {table_name}(user_id, follows_users, follows_ids, retrieve_date) " \
              "VALUES (%s, %s, %s, %s);"
        for index in range(len(df.index)):
            v2 = int(df.iloc[index, 1])
            v3 = (df.iloc[index, 2])
            v4 = int(df.iloc[index, 3])
            v5 = str(df.iloc[index, 4])
            cursor.execute(sql, (v2, v3, v4, v5))
            connection.commit()
    else:
        sql = f"INSERT INTO {table_name}(username, user_id, follows_users, follows_ids, retrieve_date) " \
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
    db_close(connection)
