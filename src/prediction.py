import json
import shutil
import time

import logwrapper
from src import utils, dispatcher
from src.spert.config_reader import process_configs
from src.spert.spert_trainer import SpERTTrainer
from src.spert import input_reader


# defines the thread for online training
def pred_thread(que):
    # Que empty? wait 2 seconds before looking again
    while True:
        if que.empty():
            time.sleep(2)
        # else train with first que element
        else:
            # get prediction id from the que
            pred_id = que.get()

            logwrapper.info(f'Starting the prdiction of {pred_id}')
            pred(pred_id)


def pred(pred_id):
    project_id, doc_id = utils.preprocess_pred_data(pred_id)
    arg_parser = utils.get_pred_args(project_id, pred_id, doc_id)

    process_configs(target=__predict, arg_parser=arg_parser)

    with open(f'pred/{pred_id}/results.json') as f:
        data = json.load(f)

    ser_data = utils.serialize_prediction(data)
    ser_data['project_id'] = project_id
    ser_data['document_id'] = doc_id

    logwrapper.info(f'Predicted {len(ser_data["recEntities"])} entities and {len(ser_data["recRelations"])} relations '
                    f'for document {doc_id} of project {project_id}')

    dispatcher.send_message(ser_data)

    shutil.rmtree(f'pred/{pred_id}')


def __predict(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.predict(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                    input_reader_cls=input_reader.JsonPredictionInputReader)
