import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import time

from vkusmart.config import config
from vkusmart.providers import OfflineVideoProvider, OneStreamVideoProvider
from vkusmart.utils import get_video_info

torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED


from collections import defaultdict
import SharedArray as sa
import pika
import json
from prometheus_client import Summary, CollectorRegistry, push_to_gateway, Gauge
from datetime import datetime
import socket

class ImgLoader():
    def __init__(self):
        """
        Constructor.
        """

        self.module_name = 'ImgLoader'                 

        self.start_time = datetime.now()
        self.registry = CollectorRegistry()
        self.cycle_time = Summary('module_cycle_time', 'Module work time', ['module', 'name'], registry=self.registry)
        self.last_success = Gauge('module_last_success_time', 'Time of las successful cycle finish', ['module', 'name'], registry=self.registry)

        ''' Frame Provider '''
        videos_provider_info = ('/videos', '03.mp4') # move to config param 
                                                                                                               # or _init_ function param
        self.dir_name = videos_provider_info[0]
        self.wildcard = videos_provider_info[1]

        conn_params = pika.ConnectionParameters('rabbit', 5672)
        self.connection = pika.BlockingConnection(conn_params)
        self.channel  = self.connection.channel()


        ##############  video provider stuff, to be cleaned ###############       
        config.DIRS.VIDEOS_DIR = Path(self.dir_name).resolve()
        config.INPUT.WILDCARD = self.wildcard
        video_wh = defaultdict()
        num_cameras = 0
        
        for video_path in config.DIRS.VIDEOS_DIR.glob(config.INPUT.WILDCARD):
            num_cameras += 1
            input_info = get_video_info(video_path)
            v_name = input_info['path'].split('/')[-1]
            video_wh[v_name] = {'height': input_info['height'], 'width': input_info['width']}
            
        videos_provider = OfflineVideoProvider(
            data_dir=config.DIRS.VIDEOS_DIR,
            wildcard=config.INPUT.WILDCARD
        )
        
        self.batch_generator = videos_provider.frames() 
        ##################################################################

        self.t0 = 0
        self.t1 = 0

    def log_time(self, msg='', from_start=False, refresh_time=True):
        if from_start:
            t = self.t0
        else:
            t = self.t1
        #print(json.dumps({'module': self.module_name, 'message': msg, 't': str(time.time()), 'dt': str(time.time() - t)}))
        self.channel.basic_publish(exchange='', routing_key='time_logs'
            , body=json.dumps({'module': self.module_name, 'message': msg, 't': str(time.time()), 'dt': str(time.time() - t)}))

        if refresh_time:
            self.t1 = time.time()

    def run(self):
        """
        # TODO: write description
        """    
        try:
            self.t0 = time.time()
            self.t1 = self.t0
            q = self.channel.queue_declare(queue='detector')
            self.channel.queue_declare(queue='time_logs')
            if q.method.message_count >= 59:
                time.sleep(1)


            frame_num, timestamp, images_list  = self.batch_generator.__next__()
            self.log_time("Took next batch:")

            sh_mem_adress = f"shm://{self.module_name}_{frame_num}"
            try:
                shared_mem = sa.create(sh_mem_adress, np.shape(images_list))
            except:
                sa.delete(sh_mem_adress)
                shared_mem = sa.create(sh_mem_adress, np.shape(images_list))
            self.log_time('Created shared memory')
            shared_mem[:] = np.array(images_list)
            self.log_time('Copied to shared memory:')
            self.channel.basic_publish(exchange='', routing_key='detector', body=sh_mem_adress)

            self.log_time('Published message:')
        ########################################################################
            del frame_num, timestamp, images_list

            self.log_time('Full time:', from_start=True)
        except StopIteration: # no more frames left in videos_provider
            print('stop iter')

            

if __name__ == "__main__":

    loader = ImgLoader()

    for i in range(51):
        with loader.cycle_time.labels(module=loader.module_name, name=socket.gethostname()).time():
            loader.run()
            push_to_gateway('pushgateway:9091', job='Test '+str(loader.start_time), registry=loader.registry)

    loader.connection.close()

