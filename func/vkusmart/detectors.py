from typing import Union, Tuple, List
#import torch.nn as nn
import numpy as np
#import tensorflow as tf

from .types import Images, Image, BoundingBoxes, BoundingBoxesWithScores
from . import object_detectors


class Detector(object):
    def predict(self, imgs: Union[Images, Image]) -> List[BoundingBoxes]:
        '''Predicts bounding boxes for each image given

        Args:
            imgs: one image or list of images
        Returns:
            List of lists of bounding boxes (which is tuple of 4 floats).
            len(result) == len(imgs)
            Length of each item of result equals count of detected persons

        TODO fix format of BBs (xywh or something else)
        '''
        pass
    
    def predict_with_scores(self, imgs: Union[Images, Image]) -> List[BoundingBoxesWithScores]:
        '''Predicts bounding boxes and their scores for each image given

        Args:
            imgs: one image or list of images
        Returns:
            List of lists of bounding boxes (which is tuple of 4 floats) and their scores (1 float)
            len(result) == len(imgs)
            Length of each item of result is equal to amount of detected persons

        TODO fix format of BBs (xywh or something else)
        '''
        pass


class YOLOv3(Detector):
    def __init__(
        self,
        framework,
        class_names,
        config_path,
        weights_path,
        img_size,
        conf_thresh,
        iou_thresh,
        gpu_num,
    ):  
        if framework == 'pytorch':
            self.detector = object_detectors.create(
                name='yolov3_pytorch',
                config_path=config_path,
                weights_path=weights_path,
                img_size=img_size,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh,
                gpu_num=gpu_num,
            )
#            self.detector = nn.DataParallel(self.detector)
#        elif framework == 'tf':
#            config = tf.ConfigProto()
#            config.gpu_options.allow_growth = True
#             config.allow_soft_placement = False
#             config.gpu_options.per_process_gpu_memory_fraction = 0.02
#            sess = tf.Session(config=config)
#            self.detector = object_detectors.create(
#                name='yolov3_tf',
#                session=sess,
#                class_names=class_names,
#                weights_file=weights_path,
#                img_size=img_size,
#                conf_thresh=conf_thresh,
#                iou_thresh=iou_thresh,
#                gpu_num=gpu_num,
#            )
#        else: print('Unknown framework')
#         print(
#             'Successfully loaded pretrained YOLOv3-{} weights from {}'.format(
#                 framework, weights_path
#             )
#         )

        
    def predict(self, imgs: Union[Images, Image]) -> List[BoundingBoxes]:
        '''see :py:funct:`.Detector.predict`'''
        if not isinstance(imgs, tuple): imgs = (imgs)
        detections_dict = self.detector.predict_multiple(imgs, return_scores=False)
        print('detections_dict==', detections_dict)
        return [detections_dict[k] for k in detections_dict] # key==cam_id
    
    
    def predict_with_scores(self, imgs: Union[Images, Image]) -> List[BoundingBoxesWithScores]:
        '''see :py:funct:`.Detector.predict_with_scores`'''
        if not isinstance(imgs, tuple): imgs = (imgs)
        detections_dict = self.detector.predict_multiple(imgs, return_scores=True)
        
#         for k in detections_dict:
#             print(k, detections_dict[k])
        
        return [detections_dict[k] for k in detections_dict] # key==cam_id
    
    
    def predict_batch(self, imgs: Union[Images, Image]) -> List[BoundingBoxes]:
        '''Predicts bounding boxes for a batch of images

        Args:
            imgs: batch of the np.array's (N, H, W, C) in RGB format
        Returns:
            List of lists of bounding boxes (which is tuple of 4 floats)
            len(result) == len(imgs)
            Length of each item of result is equal to amount of detected persons
        '''
        detected_boxes = self.detector.predict_batch(imgs)
        return detected_boxes
