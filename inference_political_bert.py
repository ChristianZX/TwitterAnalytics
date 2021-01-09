import collections
import logging
import os
import pandas as pd
from simpletransformers.classification import ClassificationModel
import tarfile
import time
import torch
from tqdm import tqdm
from transformers import logging
import TwitterAPI
import db_functions
from helper_functions import lang_detect


def unpack_model(model_name=''):
  tar = tarfile.open(f"{model_name}.tar.gz", "r:gz")
  tar.extractall()
  tar.close()


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


def bert_predictions(tweet, model):
    """
    Bert Inference for prediction.
    :param tweet: dataframe with tweets
    :param model: Bert Model
    :return: list of pr
    """

    #try:
    predictions, raw_outputs = model.predict(tweet)
    auswertung = collections.Counter(predictions)
    # except:
    #     return [[0, 0]]
    return auswertung

def init():
#if __name__ == '__main__':
    # logger = logging.getLogger("wandb")
    # logger.setLevel(logging.WARNING)

    os.environ['WANDB_MODE'] = 'dryrun'
    logging.set_verbosity_warning()

    # run()
    #unpack_model('germeval-distilbert-german')

    # define hyperparameter
    train_args = {"reprocess_input_data": True,
                  "fp16": False,
                  "num_train_epochs": 11,
                  "overwrite_output_dir": True}

    #model = ClassificationModel("bert", "distilbert-base-german-cased", num_labels=2, args=train_args)

    dir = r"C:\Users\Admin\PycharmProjects\untitled\outputs\political_bert_1605094936.6519241\checkpoint-15000"
    #dir = "F:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    model = ClassificationModel("bert", dir, num_labels=2, args=train_args)

    print(model.device)
    return model


def eval_bert():
    """
    Runs Evaluation against Evaluation accounts in table "eval_table"
    Return a printout of the results
    :return: none
    """
    data = []
    df_pred_data = pd.DataFrame(data, columns=['screen_name', 'pol', 'unpol', 'pol_time', 'left', 'right', 'lr_time'])
    sql = "select distinct username from eval_table"
    df = db_functions.select_from_db(sql)

    print("Loading BERT")
    model = init()
    print("Querying BERT")

    for index, element in tqdm(df.iterrows(), total=df.shape[0]):

        screen_name = element[0]

        df_tweets = TwitterAPI.API_tweet_multitool(screen_name, 'temp', pages=1, method='user_timeline', append=False,write_to_db=False)  # speichert tweets in DF
        if isinstance(df_tweets, str): #if df_tweets is a string it contains an error message
            continue
        start_time = time.time()
        sprache = lang_detect(df_tweets, update=True)
        runtime = time.time() - start_time
        print (f"Runtime Lang Detect: {runtime}")
        if sprache == 'nicht_deutsch':
            continue
        #df_tweets = df_tweets.iloc[0:10, :] #send only 10 tweets to Bert for precition
        start_time = time.time()
        prediction_result = []
        prediction_result.append(bert_predictions(df_tweets['tweet'], model))
        runtime = time.time() - start_time
        print (f"Runtime Bert: {runtime}")
        #try:
        #print (element)
        #auswertung = collections.Counter(prediction_result[0])
        auswertung = prediction_result[0]
        #except:
        #except TweepError as e:
             #auswertung = 0
        df_pred_data.at[index, 'screen_name'] = screen_name
        try:
            df_pred_data.at[index, 'left'] = auswertung[0]
            df_pred_data.at[index, 'right'] = auswertung[1]
        except:
            df_pred_data.at[index, 'left'] = 0
            df_pred_data.at[index, 'right'] = 0
        df_pred_data.at[index, 'lr_time'] = runtime

    #print (df_pred_data)
    print("screen_name,Pol,Unpol,Pol_Time,Left,Right,LR_Time")
    for index, element in df_pred_data.iterrows():
        print (f"{element['screen_name']},{element['pol']},{element['unpol']},{element['pol_time']},{element['left']},{element['right']},{int(element['lr_time'])}")


