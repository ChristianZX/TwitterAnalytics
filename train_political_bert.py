import pandas as pd
import os
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import f1_score, accuracy_score
import tarfile
import torch
from torch.utils.tensorboard import SummaryWriter
from time import time
import db_functions


class_list = ['LEFT', 'RIGHT']  # left = 0, right = 1


def prepare_training_data(sql_right, sql_left, sql_right_eval, sql_left_eval):
    #ToDo: Add Docstinrg
    """
    :param sql_right:
    :param sql_left:
    :return:
    """
    eval_set_limit = 10000
    # Define data sources in DB
    df_right = db_functions.select_from_db(sql_right)
    df_left = db_functions.select_from_db(sql_left)

    df_right_eval = db_functions.select_from_db(sql_right_eval)
    df_left_eval = db_functions.select_from_db(sql_left_eval)
    df_right_eval = df_right_eval.sample(frac=1)
    df_left_eval = df_left_eval.sample(frac=1)
    df_right_eval=df_right_eval.iloc[0:eval_set_limit,]
    df_left_eval=df_left_eval.iloc[0:eval_set_limit,]

    df_right['pred_class'] = 1
    df_left['pred_class'] = 0
    df_right_eval['pred_class'] = 1
    df_left_eval['pred_class'] = 0

    df = pd.concat([df_right, df_left])
    df['tweet'] = df['tweet'].str.replace('\r', "")
    df = df.sample(frac=1)

    df_eval = pd.concat([df_right_eval, df_left_eval])
    df_eval['tweet'] = df_eval['tweet'].str.replace('\r', "")
    df_eval = df_eval.sample(frac=1)

    train_df, test_df = train_test_split(df, test_size=0.10)
    print('train shape: ', train_df.shape)
    print('test shape: ', test_df.shape)
    return train_df, test_df, df_eval


def load_model(model_type: str, model_name: str, train_args: dict = None):
    # TODO: add docstring

    model = ClassificationModel(
        model_type, model_name,
        # "bert", "F:\AI\outputs\political_bert_1605477695.1526215\checkpoint-270000-epoch-11",  # #for training from checkpoint
        num_labels=2,
        args=train_args
    )
    return model


def pack_model(model_path='', file_name=''):
    files = [files for root, dirs, files in os.walk(model_path)][0]
    with tarfile.open(file_name + '.tar.gz', 'w:gz') as f:
        for file in files:
            f.add(f'{model_path}/{file}')


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


def run_BERT_training(model, train_df, test_df, eval_df):
    writer = SummaryWriter('runs')
    run()
    model.train_model(train_df, eval_dataset=eval_df)

    def f1_multiclass(labels, preds):
        return f1_score(labels, preds, average='micro')
    
    result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=f1_multiclass, acc=accuracy_score)
    
    writer.flush()
    writer.close()
    print(result)
    print(model_outputs)
    print(wrong_predictions)
    
    pack_model('outputs', 'germeval-distilbert-german_LR')


if __name__ == '__main__':
    output_dir = rf"F:/AI/outputs/political_bert_{time()}"
    train_args = {
        "reprocess_input_data": True,
        "fp16": False,
        "num_train_epochs": 30,
        "overwrite_output_dir": True,
        "save_model_every_epoch": True,
        "save_eval_checkpoints": True,
        "learning_rate": 5e-7,  # default 5e-5
        "save_steps": 5000,
        "eval_steps": 5000,
        "output_dir": output_dir,
        "warmup_steps": 2000,
        "do_eval": True,
        "per_device_train_batch_size": 12,
        "per_device_eval_batch_size": 12,
        "best_model_dir": output_dir + "/best_model/"
    }
    model_type = "bert"
    #model_name_or_path = "distilbert-base-german-cased" #new training
    model_name_or_path = "F:\AI\outputs\political_bert_1612476777.7964592\checkpoint-345000" #bert 2b
    #model_name_or_path = "F:\AI\outputs\political_bert_1605652513.149895\checkpoint-485000" #German Political Bert 1
    #model_name_or_path = "F:\AI\outputs\political_bert_1612304263.269106\checkpoint-270000-epoch-11" #German Political Bert 2 (Better BERT)

    #sql_right = "select tweet from bert2_training_left"
    #sql_left = "select tweet from bert2_training_left"

    sql_right = "select tweet from right_tweets2"
    sql_left = "select tweet from left_tweets2"


    sql_right_eval = "select tweet from (select * from right_tweets except select * from bert2_training_right) a where substring(tweet,0,3) <> 'RT'"
    sql_left_eval = "select tweet from (select * from left_tweets except select * from bert2_training_left) a where substring(tweet,0,3) <> 'RT'"

    train_df, test_df, eval_df = prepare_training_data(sql_right, sql_left, sql_right_eval, sql_left_eval)
    model = load_model(model_type, model_name_or_path, train_args)
    run_BERT_training(model, train_df, test_df, eval_df)

