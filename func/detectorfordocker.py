import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import time

from vkusmart.config import config
from vkusmart import detectors

torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED

from collections import defaultdict
import SharedArray as sa
import pika
import traceback, sys
import json
from prometheus_client import Summary, CollectorRegistry, push_to_gateway, Gauge
from datetime import datetime
import socket
###########################################################################################################################################


class Detector():
    def __init__(self):
        """
        Constructor.
        
        Attributes:
            module_name (str)                       : module name
            ...
        """
        #PipelineModule.__init__(self, pipe_registry)
        self.module_name = 'Detector'

        self.start_time = datetime.now()
        self.registry = CollectorRegistry()
        self.cycle_time = Summary('module_cycle_time', 'Module work time', ['module', 'name'], registry=self.registry)
        self.last_success = Gauge('module_last_success_time', 'Time of las successful cycle finish', ['module', 'name'], registry=self.registry)
        


        conn_params = pika.ConnectionParameters('rabbit', 5672)
        self.connection = pika.BlockingConnection(conn_params)
        self.channel  = self.connection.channel()
        
        '''
        Detector model init
        '''
        # make fabric method for this
        if config.DETECTOR.ARCH == 'yolov3':
            self.detector = detectors.YOLOv3(
                framework=config.DETECTOR.FRAMEWORK,
                class_names=config.DETECTOR.NAMES_PATH,
                config_path=config.DETECTOR.CONFIG_PATH,
                weights_path=config.DETECTOR.WEIGHTS_PATH,
                img_size=config.DETECTOR.IMG_SIZE,
                conf_thresh=config.DETECTOR.CONF_THRESH,  # may move to method args later
                iou_thresh=config.DETECTOR.IOU_THRESH,  # may move to method args later
                gpu_num=config.DETECTOR.GPU_NUM
            )
#            self.detector = nn.DataParallel(self.detector)

        # For time logging
        self.t0 = 0
        self.t1 = 0

    def log_time(self, msg='', from_start=False, refresh_time=True):
        '''
            Time logging function
        '''
        if from_start:
            t = self.t0
        else:
            t = self.t1

        self.channel.basic_publish(exchange='', routing_key='time_logs'
            , body=json.dumps({'module': self.module_name, 'message': msg, 't': str(time.time()), 'dt': str(time.time() - t)}))

        if refresh_time:
            self.t1 = time.time()
    
    def callback(self, method, body):
      with self.cycle_time.labels(module=self.module_name, name=socket.gethostname()).time():
        self.t0 = time.time()
        self.t1 = self.t0
        torch.cuda.set_device(np.random.randint(10%3))

        message = body.decode()
        if message == 'END':
            self.channel.basic_publish(exchange='', routing_key='reid', body=body)
            return

        frame_num = map(int, message.split('_')[-1]) # takes frame_num from adress


        images_list = sa.attach(message)
        self.log_time("Read image from shm:")

        bboxes = self.detector.predict_with_scores(images_list)
        self.log_time("Detector predicted:")

        bboxes = np.array([tensor[0].numpy() for tensor in bboxes[0]])
        self.log_time("Detector output into array converted:")

        if bboxes.shape[0] != 0:
            sh_mem_adress = f"shm://{self.module_name}_{frame_num}"
            try:
                shared_mem = sa.create(sh_mem_adress, bboxes.shape)
            except:
                sa.delete(sh_mem_adress)
                shared_mem = sa.create(sh_mem_adress, bboxes.shape)
            self.log_time("Shared memory created:")
            
            # copy image to shared memory
            shared_mem[:] = np.array(bboxes)
            self.log_time("Detector copied to shared memory:")

            sa.delete(message)
            sa.delete(sh_mem_adress)

            del images_list, bboxes
            torch.cuda.empty_cache()

        self.channel.basic_ack(delivery_tag = method.delivery_tag)
        self.log_time('Full time:', from_start=True)
        self.last_success.labels(module=self.module_name, name=socket.gethostname()).set_to_current_time()
        #with self.cycle_time.labels(module=self.module_name, name=socket.gethostname()).time():
        push_to_gateway('pushgateway:9091', job='Test '+str(self.start_time), registry=self.registry)

    def run(self):
        self.channel.queue_declare(queue='detector')
        self.channel.queue_declare(queue='reid')
        self.channel.queue_declare(queue='time_logs')


        self.channel.basic_consume('detector', lambda ch, method, properties, body: self.callback( method, body))

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.channel.stop_consuming()
        except Exception:
            self.channel.stop_consuming()
            traceback.print_exc(file=sys.stdout)


        


###########################################################################################################################################


if __name__ == "__main__":

    detector = Detector()
    time.sleep(1.5)
    detector.run()

    detector.connection.close()
