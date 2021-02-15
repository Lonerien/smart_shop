#!/usr/bin/env python
import pika
import traceback, sys
import numpy
#import SharedArray as sa
import time
import json

conn_params = pika.ConnectionParameters('rabbit', 5672)
connection = pika.BlockingConnection(conn_params)
channel = connection.channel()

channel.queue_declare(queue='reid')

print("Waiting for messages. To exit press CTRL+C")

def callback(ch, method, properties, body):
    body = json.loads(body)
    #address = body['address']
    i = int(body['value'])
    #b = sa.attach(address)
    #b[i] = 2
    if i % 10000 == 0: print('Reid', i)
    ch.basic_ack(delivery_tag = method.delivery_tag)

channel.basic_consume('reid', callback)

try:
    channel.start_consuming()
except KeyboardInterrupt:
    channel.stop_consuming()
except Exception:
    channel.stop_consuming()
    traceback.print_exc(file=sys.stdout)
