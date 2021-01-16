import collections
import logging
import os
import pandas as pd
from simpletransformers.classification import ClassificationModel
import time
from tqdm import tqdm
from transformers import logging
import TwitterAPI
import db_functions
from helper_functions import lang_detect


def bert_predictions(tweet: pd.DataFrame, model: ClassificationModel):
    """
    Bert Inference for prediction.
    :param tweet: dataframe with tweets
    :param model: Bert Model
    :return: list of pr
    """
    predictions, raw_outputs = model.predict(tweet)
    auswertung = collections.Counter(predictions)
    return auswertung


def load_model(model_name):
    os.environ['WANDB_MODE'] = 'dryrun'
    logging.set_verbosity_warning()

    # define hyperparameter
    # train_args = {"reprocess_input_data": True,
    #               "fp16": False,
    #               "num_train_epochs": 11,
    #               "overwrite_output_dir": True}

    #model = ClassificationModel("bert", model_name, num_labels=2, args=train_args)
    model = ClassificationModel("bert", model_name, num_labels=2)
    
    print(model.device)
    return model


def eval_bert() -> None:
    """
    Runs evaluation against evaluation accounts in table "eval_table"
    Return a printout of the results
    :return: none
    """
    data = []
    df_pred_data = pd.DataFrame(data, columns=['screen_name', 'pol', 'unpol', 'pol_time', 'left', 'right', 'lr_time'])
    sql = "select distinct username from eval_table"
    df = db_functions.select_from_db(sql)
    
    print("Loading BERT")
    #older version
    #model_path = r"C:\Users\Admin\PycharmProjects\untitled\outputs\political_bert_1605094936.6519241\checkpoint-15000"
    #model_path = r"F:\AI\outputs\political_bert_1605652513.149895\checkpoint-480000"
    model = load_model(model_path)
    print("Querying BERT")
    
    for index, element in tqdm(df.iterrows(), total=df.shape[0]):
        
        screen_name = element[0]
        
        df_tweets = TwitterAPI.API_tweet_multitool(screen_name, 'temp', pages=1, method='user_timeline', append=False,
                                                   write_to_db=False)  # speichert tweets in DF
        if isinstance(df_tweets, str):  # if df_tweets is a string it contains an error message
            continue
        start_time = time.time()
        german_language = lang_detect(df_tweets, update=True)
        runtime = time.time() - start_time
        print(f"Runtime Lang Detect: {runtime}")
        if german_language is False:
            continue
        start_time = time.time()
        prediction_result = [bert_predictions(df_tweets['tweet'], model)]
        runtime = time.time() - start_time
        print(f"Runtime Bert: {runtime}")

        result = prediction_result[0]
        df_pred_data.at[index, 'screen_name'] = screen_name
        try:
            df_pred_data.at[index, 'left'] = result[0]
            df_pred_data.at[index, 'right'] = result[1]
        except:
            df_pred_data.at[index, 'left'] = 0
            df_pred_data.at[index, 'right'] = 0
        df_pred_data.at[index, 'lr_time'] = runtime

    print("screen_name,Pol,Unpol,Pol_Time,Left,Right,LR_Time")
    for index, element in df_pred_data.iterrows():
        print(
            f"{element['screen_name']},{element['pol']},{element['unpol']},{element['pol_time']},{element['left']},"
            f"{element['right']},{int(element['lr_time'])}")
