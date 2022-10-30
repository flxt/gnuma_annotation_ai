import json
import os
import uuid

from flask import request
from flask_restful import Resource, abort

import logwrapper


# Endpoint for sending online learning requests
class Training(Resource):
    # init the resource
    def __init__(self, que, lock):
        self._que = que
        self._lock = lock

    def post(self):
        data = request.json

        # check if every thing is here
        if any(ele not in data for ele in ['project_id', 'train_data', 'entity_types', 'relation_types']):
            abort(400, message=f'Parameter missing in json.')

        # check types
        assert isinstance(data['project_id'], str)
        assert isinstance(data['train_data'], list)
        assert isinstance(data['entity_types'], list)
        assert isinstance(data['relation_types'], list)

        # generate uuid for this training rquest
        # needed to have a difference between request for same project
        new_id = str(uuid.uuid4())
        project_id = data['project_id']

        # Save data to disk
        os.makedirs(f'./train/{new_id}', exist_ok=False)

        with open(f'./train/{new_id}/config.json', 'w') as file:
            json.dump(data, file)


        # Add project id to que if not there and add to training info
        # Lock used to ensure no adding / removing at the same time
        # Because of this all que and training_info.json operations are in the lock
        # No deadlocks should be possible
        self._lock.acquire()

        with open('training_info.json', 'r') as file:
            data = json.load(file)

        # data has key project-id => append new id
        # no need to put project id in que
        if project_id in data:
            print('exists')
            data[project_id].append(new_id)
        # else create new dict element and put in que
        else:
            print('new')
            data[project_id] = [new_id]
            self._que.put(project_id)

        with open('training_info.json', 'w') as file:
            json.dump(data, file)

        self._lock.release()

        logwrapper.info(f'Put training {new_id} for project {project_id} in que.')


# Endpoint for sending online learning requests
class Prediction(Resource):
    # init the resource
    def __init__(self, que):
        self._que = que

    def post(self):
        data = request.json

        # check if every thing is here
        if any(ele not in data for ele in ['project_id', 'document_id', 'entity_types', 'relation_types']):
            abort(400, message=f'Parameter missing in json.')

        # check types
        assert isinstance(data['project_id'], str)
        assert isinstance(data['document_id'], str)
        assert isinstance(data['entity_types'], list)
        assert isinstance(data['relation_types'], list)

        # put in que if model for project exists
        if os.path.isdir(f'models/{data["project_id"]}') and len(os.listdir(f'models/{data["project_id"]}')) != 0:
            # generate uuid for this training rquest
            # needed to have a difference between request for same project
            new_id = str(uuid.uuid4())

            # Save data to disk
            os.makedirs(f'./pred/{new_id}', exist_ok=False)

            with open(f'./pred/{new_id}/config.json', 'w') as file:
                json.dump(data, file)

            self._que.put(new_id)

            logwrapper.info(f'Put prediction {new_id} for project {data["project_id"]} in que.')
