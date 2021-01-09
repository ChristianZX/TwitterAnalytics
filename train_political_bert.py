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

"""
Runs German Bert Training with data retrieved from DB
Below, Left and Right are used. Correctly it should be moderate and right_wing_populists
Setup Step 1: define checkpoint directory (row 19) 
Setup Step 2: define data sources in DB (row 22+23)
"""
# TODO: Move to main block/parameter of run()
# Checkpoint Directory
dir = rf"F:/AI/outputs/political_bert_{time()}"

# TODO: Function prepare_training_data (or similar)
# Define data sources in DB
df_rechts = db_functions.select_from_db("select tweet from s_h_afd_core_timelines limit 100000")
df_links = db_functions.select_from_db("select tweet from s_h_gruene_core_timelines limit 100000")

writer = SummaryWriter('runs')
class_list = ['LEFT', 'RIGHT']  # left = 0, right = 1

RIGHT_lst = [1 for element in range(len(df_rechts))]
LEFT_lst = [0 for element in range(len(df_rechts))]

df_rechts = df_rechts.join(pd.Series(RIGHT_lst, name='pred_class'))
df_links = df_links.join(pd.Series(LEFT_lst, name='pred_class'))

df = pd.concat([df_rechts, df_links])
df['tweet'] = df['tweet'].str.replace('\r', "")
df = df.sample(frac=1)

train_df, test_df = train_test_split(df, test_size=0.10)

print('train shape: ', train_df.shape)
print('test shape: ', test_df.shape)


# TODO: Function load_model(model_name: str, train_args: dict = None)
# TODO: How is this load_model different from the load_model in inference_political_bert.py?
train_args = {
    "reprocess_input_data": True,
    "fp16": False,
    "num_train_epochs": 30,
    "overwrite_output_dir": True,
    "save_model_every_epoch": True,
    "save_eval_checkpoints": True,
    "learning_rate": 5e-7,  # default 5e-5
    "save_steps": 5000,
    "output_dir": dir,
    "warmup_steps": 2000,
    "best_model_dir": dir + "/best_model/"
}

model = ClassificationModel(
    "bert", "distilbert-base-german-cased",
    # "bert", "F:\AI\outputs\political_bert_1605477695.1526215\checkpoint-270000-epoch-11", #for training from
    # checkpoint
    num_labels=2,
    args=train_args
)


def pack_model(model_path='', file_name=''):
    files = [files for root, dirs, files in os.walk(model_path)][0]
    with tarfile.open(file_name + '.tar.gz', 'w:gz') as f:
        for file in files:
            f.add(f'{model_path}/{file}')


# TODO: confusing naming - run() should be more specific as `run_BERT_training` also exists
# TODO: This run function exists exactly the same way in inference_political_bert.py. Write it only once and load it.
#  ==> there should be one train_model function that both files load / or no training function in the inference file
def run():
    torch.multiprocessing.freeze_support()
    print('loop')


def run_BERT_training():
    run()
    model.train_model(train_df)
    
    # TODO: Why must this function only exist inside run_BERT_training?
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
    run_BERT_training()
