import db_functions
# TODO: Integrate into actual setup.py (last step before finishing readme)
#Creates table n_users
table_name = "n_users"
sql_table = f"""
CREATE TABLE public.{table_name}
(
    index integer NOT NULL DEFAULT nextval('n_users_index_seq'::regclass),
    id bigint,
    name character varying COLLATE pg_catalog."default",
    screen_name character varying COLLATE pg_catalog."default",
    location character varying COLLATE pg_catalog."default",
    profile_location character varying COLLATE pg_catalog."default",
    followers_count character varying COLLATE pg_catalog."default",
    friends_count bigint,
    listed_count bigint,
    created_at character varying COLLATE pg_catalog."default",
    favourites_count bigint,
    verified character varying COLLATE pg_catalog."default",
    statuses_count bigint,
    last_seen character varying COLLATE pg_catalog."default",
    score_sum bigint,
    opinion_count bigint,
    ex_score_aut numeric,
    n_friends_count bigint,
    lr text COLLATE pg_catalog."default",
    lr_conf numeric,
    pol text COLLATE pg_catalog."default",
    pol_conf numeric,
    lr_pol_last_analysed text COLLATE pg_catalog."default",
    result_bert_friends text COLLATE pg_catalog."default",
    bert_friends_conf numeric,
    bert_friends_last_seen text COLLATE pg_catalog."default",
    bf_left_number integer,
    bf_right_number integer,
    combined_rating text COLLATE pg_catalog."default",
    combined_conf numeric,
    private_profile boolean,
    batch integer,
    bert_friends_ml_result text COLLATE pg_catalog."default",
    bert_friends_ml_conf numeric,
    bert_friends_ml_count integer,
    bert_friends_ml_last_seen text COLLATE pg_catalog."default"
)"""
db_functions.update_table(sql_table)

# Create table n_followers
table_name = "n_followers"
sql_table = f"""
CREATE TABLE public.{table_name}
(
    index integer NOT NULL DEFAULT nextval('n_likes_index_seq'::regclass),
    username text COLLATE pg_catalog."default",
    user_id bigint,
    follows_users text COLLATE pg_catalog."default",
    follows_ids text COLLATE pg_catalog."default",
    retrieve_date text COLLATE pg_catalog."default"
)
"""
db_functions.update_table(sql_table)

# Create table n_friends
table_name = "n_friends"
sql_table = f"""
CREATE TABLE public.{table_name}
(
    username text COLLATE pg_catalog."default",
    user_id bigint,
    follows_users text COLLATE pg_catalog."default",
    follows_ids text COLLATE pg_catalog."default",
    retrieve_date text COLLATE pg_catalog."default",
    index integer NOT NULL DEFAULT nextval('n_friends_index_seq'::regclass),
    user_id_txt text COLLATE pg_catalog."default",
    common_friends text COLLATE pg_catalog."default"
)
"""
db_functions.update_table(sql_table)

# create table facts_hastags
table_name = 'facts_hashtags'
sql_table = f"""
CREATE TABLE public.{table_name}
(
    index bigint,
    id text COLLATE pg_catalog."default",
    conversation_id text COLLATE pg_catalog."default",
    created_at bigint,
    date text COLLATE pg_catalog."default",
    tweet text COLLATE pg_catalog."default",
    hashtags text COLLATE pg_catalog."default",
    user_id bigint,
    username text COLLATE pg_catalog."default",
    name text COLLATE pg_catalog."default",
    link text COLLATE pg_catalog."default",
    retweet boolean,
    nlikes bigint,
    nreplies bigint,
    nretweets bigint,
    quote_url text COLLATE pg_catalog."default",
    user_rt_id text COLLATE pg_catalog."default",
    user_rt text COLLATE pg_catalog."default",
    from_staging_table character varying COLLATE pg_catalog."default"
)
"""
db_functions.update_table(sql_table)

# create table eval_table
table_name = 'eval_table'
sql_table = f"""
CREATE TABLE public.{table_name}
(
    index bigint,
    id bigint,
    conversation_id bigint,
    created_at text COLLATE pg_catalog."default",
    date text COLLATE pg_catalog."default",
    tweet text COLLATE pg_catalog."default",
    hashtags text COLLATE pg_catalog."default",
    user_id text COLLATE pg_catalog."default",
    username text COLLATE pg_catalog."default",
    name text COLLATE pg_catalog."default",
    link bigint,
    retweet bigint,
    nlikes bigint,
    nreplies bigint,
    nretweets bigint,
    quote_url bigint,
    user_rt_id bigint,
    user_rt bigint,
    staging_name text COLLATE pg_catalog."default",
    lr text COLLATE pg_catalog."default",
    pol text COLLATE pg_catalog."default"
)
"""
db_functions.update_table(sql_table)