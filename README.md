# BERT based Twitter Stance Analytics

The Twitter Analytics Tool can used to do political stance prediction for Twitter users.
Its findings can be used to analyse political participation in hashtags.

**Insert reference to blog post**



![Example Hashtag Anlysis][overview]

[overview]: https://github.com/ChristianZX/TwitterAnalytics/blob/feature/refactor_project/images/hashtag%20overview.PNG "Overview Image"
Visualization made with Power BI

In its current setup it will train a German BERT sequence classification model to read tweets. You can however replace
it with a BERT of a different language.
The project comes with all things you need to do your own AI-based stance analysis. Such as:

1.  PostgreSQL DWH creation
2.  Required ETL operations
3.  Download of entire hashtags
4.  Download of Twitter followers and friends
5.  BERT fine tuning
6.  BERT inference
7.  Confidence calculation

Please note that collecting your own data can take weeks, due to Twitter API download limitations. 

##Table of Content

## Setup
Set up environment
````shell script
conda create -n twlytics python=3.8
conda activate twlytics
pip install -r requirements.txt
````

Set up database
````shell script
python initial_setup.py
```` 

## Batteries not included
The projects does not come with a filled database. It will however create an empty DWH.
Therefore, it also does not come with an evaluation set. To get it, you will need to annotate
some users yourself. The more the better.  

## How it works
The main challenge in this project was to work around the limitations of the Twitter API. The result was a 
five-step process. Each step is described below:
 

![High_Level_Process][High_Level_Process]

[High_Level_Process]: https://github.com/ChristianZX/TwitterAnalytics/blob/master/images/HighLevelProcess.PNG "High Level Process"

**Step 1) Fine-Tune BERT**
Basis of the user analysis is a German BERT neural network. It was trained on German language and fine tuned
by me on political tweets. For this purpose, it read 200,000 tweets of different users, which were
previously annotated as moderate or right-wing populist. Whole accounts were annotated as right-wing
populist or moderate, not every single one of their tweets.
After this training, the artificial intelligence is able to read a German tweet and judge whether
its text is right-wing populist or moderate. Because the significance of a single tweet is generally low, the process reads 200 tweets per user and divides them into moderate and right-wing. 
![BERT_Training][BERT_training]

[BERT_Training]: https://github.com/ChristianZX/TwitterAnalytics/blob/master/images/BertTraining.PNG "BERT Training"

**Step 2) Download 200 Million Followers**
The mass of Twitter accounts has relatively few followers, while celebrities or well-known politicians have many.
In step 2 I downloaded 200 million followers from about 3,000 popular accounts. For example,
leading green party politician Annalena Baerbock, who (as of January 2021) has 108,000 followers.

**Step 3) Read Friend Tweets**
The German BERT AI reads 200 tweets from each of the 3,000 accounts and forms an opinion whether
the account is moderate or right-wing. In the case of Ms. Baerbock, it classifies 184 tweets as
moderate and 16 as right-wing. This leads to a confidence value of 84% (200/200 = 100%, 100/200 = 0%).
This is a fairly high confidence and earns her 365th place out of 24,199 accounts that got a moderate
personal rating.

**Step 4) Read the Average Users' Tweets**
Reading and classifying tweets is possible for any account that has enough German tweets (at least 75%).
Let’s imagine the account of Max Mustermann (the german John Doe) to be moderate after this first analysis,
with a confidence of 70%.

**Step 5) Calculate Combined Score**
The process now searches for Max Mustermann in the 200 million downloaded followers of the 3,000 large accounts.
Twitter refers to the accounts found in this way as Max Mustermann's friends.
Max Mustermann is found as a follower on Ms. Baerbock and Netzpolitik. Both accounts are moderate.
Netzpolitik and Ms. Baerbock each use the combined rating, the sum of the personal rating and friend rating,
so to speak. Ms. Baerbock's personal moderate rating is combined with the rating of her friends of which 137
are moderate and 11 right-wing. Which results in a moderate classification for her friends with a
confidence of 93%. This also results in a moderate combined rating, with a confidence of
(84% * 0.65 + 93% * 0.65) of 100%. The same is done with the account of Netzpolitik.

Max Mustermann's own rating and that of his friends are also combined: It is also moderate.

Only accounts that had at least 10 friends with a rating and a confidence above 70% were included in
the above hashtag analyses. 

The following chart describes the possible cases:

![Combined_Score][Combined_Score]

[Combined_Score]: https://github.com/ChristianZX/TwitterAnalytics/blob/master/images/combined_calculation.png "Combined_Score"


**Step 6 of 5) Optional**
The 200 million followers currently available are too few to calculate a good Friend Rating for
a high percentage of users. To be able to estimate as many users of a hashtag as possible,
I download the list of their friends directly if necessary,, in contrast to the indirect approach
via the followers of e.g. Annalena Baerbock or Netzpolitik.

So if Max Mustermann follows a total of 150 accounts, I download them completely.
One result could be that Max Mustermann also follows Günther Jauch (TV anchor so popular quite
some want him to become president). Since I never downloaded the followers of Günther Jauch, there
was no connection and also no friend rating. If I got a score for ten of Max Mustermann's 150 friends,
I can calculate a friend rating for him.

The big disadvantage of this approach lies in the limitations of the Twitter API. Per minute it allows
to download the friends of one user (batch size 5,000). Since Max Mustermann only has 150 friends,
I can only download 150 in this minute.
The 5,000 per minute rule also applies to the follower download. In the case of Ms. Baerbock,
who has over 100,000 followers, the 5,000 are exhausted. This means that 300,000 followers can be
downloaded per hour.
That's why downloading friends is only a last resort.

## Usage
**Note: Parametrization via JSON config files is still work in progress**

1. Get yourself a key for the [Twitter API](https://developer.twitter.com/en/apply-for-access)
2. Install [Postgres SQL](https://www.postgresql.org/download/)
3. Configure DB API Keys and DB connection in `API_KEYS_Template.py` and `db_function.py.db_connect()` and rename
   `API_KEYS_Template.py` to `API_KEYS.py`.
4. Run `dwh_setup.py` to create tables `n_users`, `n_followers`, `n_friends`, `facts_hashtags` and `eval_table`
5. Create lists of moderate and right wing accounts. Use `TWA_main.py.download_user_timelines()` to download them.
    ````python
    download_user_timelines(political_moderate_list, right_wing_populists_list)
    ````
6. Configure `train_political_bert.py` and run BERT Training
7. After BERT Training configure `inference_political_bert`.
8. Download hashtag of your choosing
9. Copy downloaded tweets you want to keep into table facts_hashtags:
    ````sql
    insert into facts_hashtags 
    select index, id, conversation_id, 0 as created_at, date, tweet, hashtags, user_id, username,
    name, link, False as retweet, nlikes, nreplies, nretweets, quote_url, user_rt_id, user_rt, staging_name 
    from STAGING_TABLE_NAME
    ````
10. Use BERT to give each user in a hashtag a rating based on their 200 latest tweets. This will store actual results 
    for your users in DB table `n_users`. At this point the results will be much better then guessing but still have low 
    accuracy, low recall and low precision.
    ````python
    sql = "select distinct user_id, username from facts_hashtags"
    model_path = r"C:\YOUR_PATH\YOUR_CHECKPOINT"
    #Example: model_path = r"C:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    user_analyse_launcher(iterations=1, sql=sql, model_path=model_path)
    ````
11. Create an evaluation set to calculate accuracy.
    ````python
    # Annotate users and write result into column eval_table.lr
    helper_functions.add_eval_user_tweets(moderate_ids, right_ids)
    TWA_Main.eval_bert(model_path)
    ```` 
    This will download the tweets of the evaluation users into `eval_table`. Without the tweets
    you can't do consistent evaluations since the users latest 200 tweets change constantly.  
12. Download some friends. During initial setup it makes sense to download the friends of a portion of your hashtag 
    users. It allows you to identify people they commonly follow and download the followers of these accounts. 
    Consider keeping the number of friends to download low. Twitter allows only 1 download per minute.
    ````python
    sql = "select distinct user_id from facts_hashtags limit 1000"
    TWA_main.get_friends(sql)
    ```` 
13. Run follower download. The Twitter API downloads 300.000 followers per hour. You will need many millions. So this 
    will take a while. Finding good accounts to download followers from is tricky. Common friends of your hashtag users
    should be a good place to start.
    ````python
    sql = """
    select follows_ids from n_friends
    group by follows_ids
    having count(follows_ids) >= 10
    order by count(follows_ids) desc
    """
    TWA_main.get_followers(sql)
    ````
14. Configure and run friend score and combined score calculation:
    ````python
    sql = """
    SELECT DISTINCT u.id AS user_id, screen_name AS username 
    FROM n_users u," 
        (SELECT follows_ids, COUNT (follows_ids) 
        FROM n_followers f 
        GROUP BY follows_ids 
        HAVING COUNT (follows_ids) >= 100) a, 
        facts_hashtags fh
        WHERE CAST (a.follows_ids AS bigint) = u.id 
        AND lr IS NULL 
        AND fh.user_id = u.id) 
    """

    model_path = r"YOUR_MODE_PATH"
    user_analyse_launcher(iterations=1, sql=sql, model_path=model_path)
    
    bert_friends_high_confidence_capp_off = 0.70
    self_conf_high_conf_capp_off = 0.70
    min_required_bert_friend_opinions = 10
    
    sql = """
    SELECT id, screen_name, lr, lr_conf, result_bert_friends, bert_friends_conf, bf_left_number,bf_right_number
    FROM n_users
    WHERE (lr IS NOT NULL
    OR result_bert_friends IS NOT NULL)
    AND combined_rating IS NULL
    """
    combined_scores_calc_launcher(
        sql, 
        bert_friends_high_confidence_capp_off, 
        self_conf_high_conf_capp_off, 
        min_required_bert_friend_opinions
    )
    
    #Run combined_scores for users that got a friend rating after downloading their friends directly
    sql = f"""
    SELECT id, screen_name, lr, lr_conf, result_bert_friends, bert_friends_conf, bf_left_number, bf_right_number
    FROM n_users
    WHERE (lr IS NOT NULL
    OR result_bert_friends IS NOT NULL)
    AND combined_rating IS NULL
    AND id in (SELECT DISTINCT f.user_id FROM facts_hashtags f)
    """
    get_combined_scores_from_followers(sql)
    ````
    
15. Print number of newly found results
    ````python
    sql = f"""
    SELECT COUNT (DISTINCT u.id)
    FROM facts_hashtags f, n_users u
    WHERE f.user_id = u.id
    AND combined_rating in ('links','rechts')"""
    print (db_functions.select_from_db(sql))
    ````
16. Re-run evaluation vs. combined score. Below statement gives you the number of correct evaluations using a confidence cut off at 70%.  
    ````python
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

## Future Plans  
* create setup.py
* Parametrization via JSON config files is still work in progress
* Improve German BERT Training: Analyse individual tweets of each training set and use only those for training, that have a high confidence.
* If between 50% and 75% of user Tweets are german continue loading tweets until 200 are german have been downloaded.
* Train englisch BERT to give self rating to englisch account and improve Friend-Rating.
* Stop language detection if 100% of the first 50 Tweets are german, if this improves performance significantly.
* Run statistical test for hashtag analysis: Assuming moderate and right accounts have an accuracy of 90%, to what degree do classification mistakes cancel out each other during hashtag analysis?
* Learn to detect left extremists to eventually identify three classes: right, moderate, left.


 






 




 


