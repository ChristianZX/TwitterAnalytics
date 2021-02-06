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

**Step 3a) Read Friend Tweets (initial  approach)**

The German BERT AI reads 200 tweets from each of the 3,000 accounts and forms an opinion whether
the account is moderate or right-wing. In the case of Ms. Baerbock, it classifies 184 tweets as
moderate and 16 as right-wing. This leads to a confidence value of 84% (200/200 = 100%, 100/200 = 0%).
**So this approach simply counts friends in both classes.** 
This is a fairly high confidence and earns her 365th place out of 24,199 accounts that got a moderate
personal rating.
This approach will produce quite good results with a relatively few (3.000) BERT self ratings.  

**Step 3b) Read Friend Tweets (AI based approach)**

In step 3b, a Random Forest classifier can be trained. However, the training matrix consisted of the 3000 follower-rich 
accounts (as columns) and 140.000 (70k left, 70k right) users in the rows who had
* a rating with a high to very high confidence 
* followed at least 5 of the 3000 users

Accuracy of 3b is substantially better than of 3a, but it needs more and better data as input.    

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

The process is controlled by the `config.ini`. To execute a task, set its value in chapter 
`[SETUP]` or `[TASKS]` to `True`. All tasks that are set to `True` will be executed successively. 
 
1. Get yourself a key for the [Twitter API](https://developer.twitter.com/en/apply-for-access)
2. Install [PostgreSQL](https://www.postgresql.org/download/)
3. Configure DB API keys and DB connection in `API_KEYS_Template.py` and `db_function.py.db_connect()` and rename
   `API_KEYS_Template.py` to `API_KEYS.py`.
4. Run `dwh_setup.py` to create tables `n_users`, `n_followers`, `n_friends`, `facts_hashtags` and `eval_table`
5. The next step is to download training data for the BERT AI. Open `config.ini` and in `[SETUP]`, set `download_BERT_training_data = True`. In chapter `[DOWNLOAD_BERT_TRAINING_DATA]` of the ini file,
configure account names (Twitter screen names) in lists `political_moderate_list` and `right_wing_populists_list`.
Afterwards excute TWA_main.py with paramater `--name config.ini`    
6. Configure `train_political_bert.py` and run BERT training. Training is not configurable via `config.ini` and needs 
    to be run from `train_political_bert.py`. The training will run some hours and needs to be aborted manually. 
    Otherwise, it will run for the configured number of epochs.      
7. Download all tweets tagged with a hashtag by setting `download_hashtag = True`. Configuration example for download 
    settings:
    ````ini
    [GLOBAL]
    hashtag = trump 
    [DOWNLOAD_HASHTAG]
    since = 2021-01-25
    until = 2021-01-31
    ````
8. Copy downloaded tweets you want to keep into table `facts_hashtags`:
    ````sql
    insert into facts_hashtags 
    select index, id, conversation_id, 0 as created_at, date, tweet, hashtags, user_id, username,
    name, link, False as retweet, nlikes, nreplies, nretweets, quote_url, user_rt_id, user_rt, staging_name 
    from STAGING_TABLE_NAME
    ````
   This is a manual operation. The name of the staging table is returned during the process.
9. Use BERT to give each user in a hashtag a rating based on their 200 latest tweets. This will store results 
    for your users in DB table `n_users`. At this point the results will be much better than guessing but still have low 
    accuracy, low recall and low precision.
    In the ini file, set the task `calculate_BERT_self_rating = True` (and `download_BERT_training_data = True`),
    and in `[CALCULATE_BERT_SELF_RATING]`, set `model_path = path of your BERT checkpoint folder`, `iterations` 
    (divides the dataset into n batches, so use 1 for small datasets of up to 500 users, and f.i. 10, if you want to 
    split 5,000 items into 10 batches of 500). For the initial inference, I suggest to use this SQL statement instead 
    of the pre-configured one (for recurring runs) in the same section: `select u.id AS user_id, screen_name AS username from n_users`
10. Use `download_BERT_training_data` to download users and their tweets to `eval_table`. 
    Afterwards, set `rate_eval_table_accounts = True` in `config.ini` to rate these accounts. Then annotate each user manually
    in column `eval_table.lr` to determine how many results were predicted correctly.
    It's important to save the users' tweets in the `eval_table`. Without them, you can't do consistent evaluations since the users latest 200 tweets change constantly.  
12. Download some friends. During initial setup, it makes sense to download the friends of a portion of your hashtag 
    users. It allows you to identify people they commonly follow and download the followers of these accounts. 
    Consider keeping the number of friends to download low. Twitter allows only 1 download per minute.
    ````ini
    [TASKS]
    download_friends = True
    [GLOBAL]
    hashtag = trump
    [DOWNLOAD_FRIENDS]
    sql = SELECT distinct f.user_id
    FROM facts_hashtags f
    left join n_friends fr on f.user_id = fr.user_id
    left join n_users u on f.user_id = u.id
    WHERE from_staging_table like 'INSERT_HASHTAG'
    and fr.user_id is null and u.private_profile is null
    ````
13. Run follower download. The Twitter API downloads 300.000 followers per hour. You will need many millions. So this 
    will take a while. Finding good accounts to download followers from is tricky. Common friends of your hashtag users
    should be a good place to start.
    ````ini
    [TASKS]
    download_followership = True
    [GLOBAL]
    hashtag = trump
    [DOWNLOAD_FOLLOWERSHIP]
    user_ids = select follows_ids as id from
        (select fr.follows_ids, count(fr.follows_ids)
        from facts_hashtags f, n_friends fr
        where from_staging_table like 'INSERT_HASHTAG' and f.user_id = fr.user_id
        group by fr.follows_ids
        having count(fr.follows_ids) >= 500
        order by count(fr.follows_ids) desc) a
      except select cast (user_id as text) from n_followers
    download_limit = 12500000
    sql_insert =  insert into n_users (id) select distinct user_id from n_followers except select id from n_users
    ````
    Alternatively you can use a list of user_ids: `user_ids = [29180690,805308596,4052291878]`
14. Calculate BERT friend rating by following the configuration example in `CALCULATE_BERT_FRIEND_RATING` 
15. Calculate BERT ML rating as described in `[CALCULATE_ML_FRIEND_RATING]` 
16. Calculate combined score from self rating, friend rating and friend ML rating as described in `[CALCULATE_COMBINED_RATING]`


## Future Plans  
* Improve German BERT Training: Analyse individual tweets of each training set and use only those for training, that have a high confidence.
* Include a Topic Model
* If between 50% and 75% of user Tweets are german continue loading tweets until 200 are german have been downloaded.
* Train english BERT to give self rating to englisch account and improve Friend-Rating.
* Stop language detection if 100% of the first 50 Tweets are german, if this improves performance significantly.
* Run statistical test for hashtag analysis: Assuming moderate and right accounts have an accuracy of 90%, to what degree do classification mistakes cancel out each other during hashtag analysis?
* Learn to detect left extremists to eventually identify three classes: right, moderate, left.


 






 




 


