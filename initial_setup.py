import db_functions

# TODO: Integrate into actual setup.py (last step before finishing readme)

"""
Refactoring To Dos:
helper_functions.conf_value() müsste durch helper_functions.interpret_stance() zu ersetzen sein 

Things yet to be improved:

Additional points for readme:
A) Unused Feature: TFIDF_inference.py and train_political_bert.py are used to create a (Random Forrest) based stance
if an account is political or unpolitical.
However that feature currently doest not add meaningful insights, since I only use the tool to analyse political
hashtags
B) ...   

Steps still to implement for automated setup:
A) Download hashtag and save it to fact_hashtags
B) create facts_hashtags table (Wenn nicht, müssen wir die distinct User für die Hashtag Analyse aus der Staging
Tabelle nehmen)


1. Install Postgres and create a DB
2. Obtain you personal Twitter API acces from Twitter 
3. Create file API_KEYS.py and follow this structure with your own keys. Below keys are not working examples:

import psycopg2
API_KEY = "ilhsnfvykjnkjynxvökj"
API_KEY_Secret = "kjsgövlkjgölkynjvölkjvnlkndmöylkjlödsvk"
ACCESS_TOKEN = "kdjnyxgkjyxnbfynbkllynjgskdngdvkjwye"
ACCESS_TOKEN_SECRET = "liukjhvylkyjhgekjalgkjdsölgkajdsldkfjnölk"
connection_string = psycopg2.connect(user="postgres",
                              password="YOUR_PASSWORD",
                              host="127.0.0.1",
                              port="5433",
                              database="YOUR_DB")

engine_string = 'postgresql://postgres:YOUR_PASSWORD@localhost:5433/YOUR_DB'
4. Execute below code to run initial DB setup

"""
# Creates table n_users
table_name = "n_users2"
sql_table = f"""CREATE TABLE public.{table_name} (index integer NOT NULL DEFAULT nextval(
'n_users_index_seq'::regclass),    id bigint,    name character varying COLLATE pg_catalog."default",    screen_name
character varying COLLATE pg_catalog."default",    location character varying COLLATE pg_catalog."default",
profile_location character varying COLLATE pg_catalog."default",    followers_count character varying COLLATE
pg_catalog."default",    friends_count bigint,    listed_count bigint,    created_at character varying COLLATE
pg_catalog."default",    favourites_count bigint,    verified character varying COLLATE pg_catalog."default",
statuses_count bigint,    last_seen character varying COLLATE pg_catalog."default",    gen1 bigint,    gen2 bigint,
 score_sum bigint,    opinion_count bigint,    gen3 bigint,    gen1_aut bigint,    gen2_aut bigint,
 gen3_aut bigint,    score_sum_aut bigint,    opinion_count_aut bigint,    ex_score numeric,    ex_score_aut numeric,
    n_friends_count bigint,    lr text COLLATE pg_catalog."default",    lr_conf numeric,    pol text COLLATE
    pg_catalog."default",    pol_conf numeric,    lr_pol_last_analysed text COLLATE pg_catalog."default",
    result_sai text COLLATE pg_catalog."default",    percent_filled_sai numeric,    probability_sai text COLLATE
    pg_catalog."default",    last_seen_sai text COLLATE pg_catalog."default",    result_bert_friends text COLLATE
    pg_catalog."default",    bert_friends_conf numeric,    bert_friends_last_seen text COLLATE pg_catalog."default",
      bf_left_number integer,    bf_right_number integer)"""
db_functions.update_table(sql_table)

# insert first entry in n_users
db_functions.update_table(f"INSERT INTO {table_name} (id) VALUES (964129015425654786)")  # insert first account into
# n_users

# Create table n_followes
sql_table = """CREATE TABLE public.n_followers(index integer NOT NULL DEFAULT nextval('n_likes_index_seq'::regclass),
   username text COLLATE pg_catalog."default",    user_id bigint,    follows_users text COLLATE pg_catalog."default",
      follows_ids text COLLATE pg_catalog."default",    retrieve_date text COLLATE pg_catalog."default",
      private integer)"""
db_functions.update_table(sql_table)

# Create table n_friends
sql_table = """CREATE TABLE public.n_friends (username text COLLATE pg_catalog."default",    user_id bigint,
follows_users text COLLATE pg_catalog."default",    follows_ids text COLLATE pg_catalog."default",    retrieve_date
text COLLATE pg_catalog."default",    index integer NOT NULL DEFAULT nextval('n_friends_index_seq'::regclass),
private integer,    user_id_txt text COLLATE pg_catalog."default",    common_friends text COLLATE
pg_catalog."default",    CONSTRAINT n_friends_pkey PRIMARY KEY (index))"""
db_functions.update_table(sql_table)
