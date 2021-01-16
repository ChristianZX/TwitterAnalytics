# BERT based Twitter Stance Analytics

The Twitter Analytics Tool can used to do political stance prediction for Twitter users.
Its findings can be used to analyse political participation in hashtags.

**Insert reference to blog post**



![Example Hashtag Anlysis][overview]

[overview]: https://github.com/ChristianZX/TwitterAnalytics/blob/feature/refactor_project/images/hashtag%20overview.PNG "Overview Image"
Visualization done by Power BI

In it's current setup it will train a German BERT-AI to read tweets. You can however replace
it with a BERT of a different language.
The project comes with all things you need to do your own AI-based stance analysis. Such as:

1.  PostgreSQL DWH creation
2.  Required ETL operations
3.  Download of entire Hashtags
4.  Download of Twitter followers and friends
5.  BERT Machine learning training
6.  BERT Inference
7.  Confidence calculation

Please note that collecting your own data can take weeks, due to Twitter API download limitation. 

##Table of Content

## Setup
**TODO: setup.py?**

Set up environment
````shell script
conda create twlytics python=python3.8
conda activate twlytics
pip install -r requirements.txt
````

Run setup
````shell script
# python setup.py
````

Set up database
````shell script
python initial_setup.py
```` 

## Batteries not included
The projects does not come with a filled database. It will however create an empty DWH.
Therefore it also doest not come with an evaluation set. To get it, you will need to annotate
some users yourself. The more the better.  

## How it works
The main challenge in this project was to work around the limitations of the Twitter API. The result was a five step process:
 

![High_Level_Process][High_Level_Process]

[High_Level_Process]: https://github.com/ChristianZX/TwitterAnalytics/blob/master/images/HighLevelProcess.PNG "High Level Process"

**Step 1) Fine-Tune BERT**
The used German BERT needs to be fine-tuned on political tweets. I did it with 100.000 moderate
and right wing Tweets. Since I could not annotate the 200.000 Tweets I used lists of
clearly moderate and right-wing politicians. This approach haas disadvantages (low recall) which I will discus in step 5
and hope to improve in future.

![BERT_Training][BERT_training]

[BERT_Training]: https://github.com/ChristianZX/TwitterAnalytics/blob/master/images/BertTraining.PNG "BERT Training"


**Step 2) Download 200 Million Followers**

**Step 3) Read Friend Tweets**

**Step 4) Read the average Users Tweets**

**Step 5) Calculate Combined Score**

Due to the rather general Bert training (annotation per account instead of Tweet) we need to
set a high confidence threshold (70%). That reduces the number of accounts we can confidently predict.

**TALK ABOUT ACCURACY in Step 5**  
**INSERT COMBINED SCORE CALCULATION IMAGE**


## Usage
**Note: Parametrization via JSON Config Files is still in progress**

1. Get yourself a [Twitter API](https://developer.twitter.com/en/apply-for-access)
2. Install [Postgres SQL](https://www.postgresql.org/download/)
3. Configure DB API Keys and DB connection in API_KEYS_Template.py and db_function.py.db_connect()
4. Run dwh_setup.py to create tables n_users, n_followers, n_friends, facts_hashtags and eval_table
5. Create Lists of moderate and right wing accounts. Use TWA_main.py.download_user_timelines() to download them.
    ````
    download_user_timelines(political_moderate_list, right_wing_populists_list)
    ````
6. Install [Tensorboard](https://pypi.org/project/tensorboard/) to monitor BERT Training 
7. Configure train_political_bert.py **JSON FILE?** and run BERT Training
8. After BERT Training configure inference_political_bert.
9. Download hashtag of your choosing **Write Json Config File**
10. Copy downloaded tweets you want to keep into table facts_hashtags:
    ````
    insert into facts_hashtags select index, id, conversation_id, 0 as created_at, date, tweet, hashtags, user_id, username, name, link, False as retweet, nlikes, nreplies, nretweets, quote_url, user_rt_id, user_rt, staging_name from STAGING_TABLE_NAME
    ````
11. User BERT to give each Hashtag user a rating based on their 200 latest tweets. This will store actual results for your users in DB table n_users. At this point the results will be much
better then guessing but still have low accuracy, low recall and low precision.
    ````
    sql = "select distinct user_id, username from facts_hashtags"
    model_path = r"C:\YOUR_PATH\YOUR_CHECKPOINT"
    #Example: model_path = r"C:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    user_analyse_launcher(iterations=1, sql=sql, model_path=model_path)
    ````
12. Create an evaluation set to calculate accuracy.
    ````
    helper_functions.add_eval_user_tweets(moderate_ids, right_ids)
    inference_political_bert.eval_bert(model_path)
    Annotate users and write result into column eval_table.lr
    ```` 
    This will download the tweets of the evaluation users into eval_table. Without the tweets
    you can't do consistent evaluations since the users latest 200 tweets change constantly.  
13. Download some friends. During initial setup it makes sense to download the friends of a
portion your hashtag users. It allows you to identify people they commonly follow and download
the followers of accounts. Consider keeping the number of friend to download low. Twitter allows only 1 download per minute.
    ````
    sql = "select distinct user_id from facts_hashtags limit 1000"
    TWA_main.get_friends(sql)
    ```` 
14. Run Follower Download. The Twitter API downloads 300.000 followers per hour. You will need many millions. So this will take a while.
Finding good accounts to download followers from is tricky. Beginning with common friends of your hashtag users
should be a good place to start.
    ````
    sql = "select follows_ids from n_friends
           group by follows_ids
           having count(follows_ids) >= 10
           order by count(follows_ids) desc"
    TWA_main.get_followers(sql)
    ````
15. Configure and run friend score and combined score calculation:
    ````
    sql = "SELECT DISTINCT u.id AS user_id,
			screen_name AS username
            FROM n_users u,
            (SELECT follows_ids,
                  COUNT (follows_ids)
            FROM n_followers f
            GROUP BY follows_ids
            HAVING COUNT (follows_ids) >= 100) a,
             facts_hashtags fh
            WHERE CAST (a.follows_ids AS bigint) = u.id
              AND lr IS NULL
              AND fh.user_id = u.id

    model_path = r"YOUR_MODE_PATH"
    user_analyse_launcher(iterations=1, sql=sql, model_path=model_path)
    
    bert_friends_high_confidence_capp_off = 0.70
    self_conf_high_conf_capp_off = 0.70
    min_required_bert_friend_opinions = 10
    
    sql = """
    SELECT id,
       screen_name,
       lr,
       lr_conf,
       result_bert_friends,
       bert_friends_conf,
       bf_left_number,
       bf_right_number
    FROM n_users
    WHERE (lr IS NOT NULL
       OR result_bert_friends IS NOT NULL)
    AND combined_rating IS NULL
    """
    combined_scores_calc_launcher(sql, bert_friends_high_confidence_capp_off, self_conf_high_conf_capp_off, min_required_bert_friend_opinions)
    
    #Run combined_scores for users that got a friend rating after downloading their friends directly
    sql = f"""
    SELECT id,
           screen_name,
           lr,
           lr_conf,
           result_bert_friends,
           bert_friends_conf,
           bf_left_number,
           bf_right_number
    FROM n_users
    WHERE (lr IS NOT NULL
           OR result_bert_friends IS NOT NULL)
      AND combined_rating IS NULL
      AND id in
        (SELECT DISTINCT f.user_id
         FROM facts_hashtags f
    """
    get_combined_scores_from_followers(sql)
    ````
    
16. Print number of newly found results
    ````
    sql = f"""
    SELECT COUNT (DISTINCT u.id)
    FROM facts_hashtags f,
         n_users u
    WHERE f.user_id = u.id
      AND combined_rating in ('links',
                              'rechts')"""
    print (db_functions.select_from_db(sql))
    ````
17. Re run evaluation vs. combined score. Below statement gives you the number of correct evaluations using a confidence cut off of 70%.  
    ````
    sql = f"""
    select distinct e.user_id, e.username, e.lr, e.pol, u.combined_rating, u.combined_conf
    from eval_table e, n_users u
    where 1=1
    and e.lr is not null
    and e.user_id <> ''
    and u.id = cast (e.user_id as bigint)
    and u.combined_conf >= 0.7
    and e.lr = u.combined_rating
    """ 
    print (db_functions.select_from_db(sql))
    ```` 

  

 





 




 


