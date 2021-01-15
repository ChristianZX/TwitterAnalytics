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


def prepare_training_data(sql_right, sql_left):
    #ToDo: Add Docstinrg
    """

    :param sql_right:
    :param sql_left:
    :return:
    """

    # Define data sources in DB
    df_right = db_functions.select_from_db(sql_right)
    df_left = db_functions.select_from_db(sql_left)

    RIGHT_lst = [1 for element in range(len(df_right))]
    LEFT_lst = [0 for element in range(len(df_right))]

    df_right = df_right.join(pd.Series(RIGHT_lst, name='pred_class'))
    df_left = df_left.join(pd.Series(LEFT_lst, name='pred_class'))

    df = pd.concat([df_right, df_left])
    df['tweet'] = df['tweet'].str.replace('\r', "")
    df = df.sample(frac=1)

    train_df, test_df = train_test_split(df, test_size=0.10)

    print('train shape: ', train_df.shape)
    print('test shape: ', test_df.shape)

    return train_df, test_df

def load_model(model_type: str, model_name: str, train_args: dict = None):
    # TODO: add docstring

    model = ClassificationModel(
        model_type, model_name,
        # "bert", "F:\AI\outputs\political_bert_1605477695.1526215\checkpoint-270000-epoch-11",
        # #for training from checkpoint
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


def run_BERT_training(model, train_df, test_df):

    writer = SummaryWriter('runs')
    run()
    model.train_model(train_df)

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
        "output_dir": output_dir,
        "warmup_steps": 2000,
        "best_model_dir": output_dir + "/best_model/"
    }
    model_type = "bert"
    #model_name_or_path = "distilbert-base-german-cased" #new training
    model_name_or_path = "F:\AI\outputs\political_bert_1605652513.149895\checkpoint-485000" #continue training from checkpoint

    sql_right = "select tweet from s_h_afd_core_timelines limit 100000"
    sql_left = "select tweet from s_h_gruene_core_timelines limit 100000"

    train_df, test_df = prepare_training_data(sql_right, sql_left)
    model = load_model(model_type, model_name_or_path, train_args)
    run_BERT_training(model, train_df, test_df)
