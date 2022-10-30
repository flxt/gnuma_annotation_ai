import json
import os.path
import shutil
import time

import logwrapper
from src import utils
from src.spert.config_reader import process_configs
from src.spert.spert_trainer import SpERTTrainer
from src.spert import input_reader


# defines the thread for online training
def train_thread(que, lock):
    # Que empty? wait 2 seconds before looking again
    while True:
        if que.empty():
            time.sleep(2)
        # else train with first que element
        else:
            # get training data list for project
            lock.acquire()

            # get project id from the que
            project_id = que.get()

            # get the folder names for training
            with open('training_info.json', 'r') as file:
                data = json.load(file)

            ids = data.pop(project_id)

            with open('training_info.json', 'w') as file:
                json.dump(data, file)

            lock.release()

            logwrapper.info(f'Starting the training for project {project_id} for {len(ids)} online steps')
            train(project_id, ids)


def train(project_id, ids):
    for count, id in enumerate(ids):
        utils.preprocess_train_data(id)

        bool = os.path.isdir(f'models/{project_id}')
        arg_parser = utils.get_train_args(project_id, bool, id)

        process_configs(target=__train, arg_parser=arg_parser)

        shutil.rmtree(f'train/{id}')

    logwrapper.info(f'Finished training for project {project_id}')


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)