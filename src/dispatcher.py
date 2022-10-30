import json

import pika as pika
import requests

import logwrapper

# Class for dispatching messages via RabbitMQ
def send_message(data: dict):
    with open('./config.json', 'r') as f:
        config = json.load(f)

    rabbit_creds = config['rabbit_creds']

    creds = pika.PlainCredentials(rabbit_creds['user'], rabbit_creds['pw'])
    params = pika.ConnectionParameters(host=rabbit_creds['host'], port=rabbit_creds['port'], credentials=creds)

    # init connection and channel
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    exchange = rabbit_creds['exchange']

    # create exchange
    channel.exchange_declare(exchange=exchange, exchange_type='fanout', durable=True)

    channel.basic_publish(exchange=exchange, routing_key='',
                          body=json.dumps(data),
                          properties=pika.BasicProperties(headers={'type': 'ai_update'}))

    connection.close()

    logwrapper.info(f'Sent results for {data["project_id"]} via RabbitMQ.')


def get_document(doc_id):
    with open('./config.json', 'r') as f:
        config = json.load(f)

    doc_address = config['doc_address'] + '/api/v1/documents/' + doc_id

    response = requests.get(doc_address)
    response.raise_for_status()

    dat = response.json()
    return dat
