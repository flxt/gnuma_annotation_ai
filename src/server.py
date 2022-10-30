import json
import os
import sys
from threading import Lock, Thread

from flask import Flask
from flask_restful import Api
from flask_cors import CORS

from queue import Queue

from resources import Training, Prediction
from src import logwrapper
from src.prediction import pred_thread
from training import train_thread


def main():
    # read config
    if not os.path.exists('./config.json'):
        logwrapper.error('No Config File. Shutting down.')
        sys.exit()

    with open('./config.json', 'r') as f:
        config = json.load(f)

    # init flask
    app = Flask(__name__)
    api = Api(app)

    # Define CORS settings.
    cors = CORS(app, resources={
        r'/api/*': {
            'origins': '*'
        }
    })

    # define the training and prediction que
    train_que = Queue()
    pred_que = Queue()

    # define the lock
    lock = Lock()

    # Addresources to the api
    pre = '/api/v1'
    api.add_resource(Training, f'{pre}/train', resource_class_kwargs={'que': train_que, 'lock': lock})
    api.add_resource(Prediction, f'{pre}/pred', resource_class_kwargs={'que': pred_que, })

    # start the training thread
    t = Thread(target=train_thread, args=(train_que, lock))
    t.start()

    # start the training thread
    t2 = Thread(target=pred_thread, args=(pred_que, ))
    t2.start()

    # Start the server.
    app.run(debug=False, port=config['port'], host='0.0.0.0')


if __name__ == '__main__':
    main()
