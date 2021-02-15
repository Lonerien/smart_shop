# -*- coding: utf-8 -*-

import time

import numpy as np
import tensorflow as tf

from typing import List, Dict, Tuple
from collections import defaultdict

from PIL import Image

from ..detector import Detector

from .yolov3 import yolo_v3, load_weights, detections_boxes, non_max_suppression
from ..utils.utils import load_names, draw_boxes, convert_to_original_size


class YOLOv3(Detector):
    
    def __init__(
        self, 
        session,
        class_names, 
        weights_file,
        iou_thresh,
        conf_thresh,
        img_size,
        gpu_num
    ):
        '''
        "You Only Look Once (v3)" object detector instance.
        
        :param session: tf.Session() instance for model initialization
        :param class_names: dict {class_id: name, ..} or name of file with this dict
        :param weights_file: name of the file with model weights
        :param iou_threshold: IoU treshold for detector
        :param conf_threshold: confidence threshold for detector
        :param img_size: image size for the detector (an input will be resized to this size)
        :param gpu_num: index of GPU to use
        
        :return: None
        '''
        self.classes = class_names if type(class_names) == type({}) else load_names(class_names) 
        self.weights_file = weights_file
        self.session = session
        self.img_size = img_size
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

        with tf.device('/gpu:{}'.format(gpu_num)):
            print('Using GPU={}'.format(gpu_num))
            self.inputs = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, 3])
            with tf.variable_scope('detector'):
                detections = yolo_v3(
                    inputs=self.inputs, 
                    num_classes=len(self.classes), 
                    data_format='NCHW'
                )
                load_ops = load_weights(
                    var_list=tf.global_variables(scope='detector'), 
                    weights_file=self.weights_file
                )
                print('load_ops created')  
            self.boxes = detections_boxes(detections)
            self.session.run(load_ops)
            print('load_ops processed successfully')
        
        
    def fit(self):
        pass
    
    
    def predict(
        self, 
        img, 
        verbose=False,
        draw_on_image=False
    ):
        '''
        :param img: PIL.Image of shape (H, W, C) in colorspace in RGB format
        :param verbose: if True, print prediction time and other info
        
        :return
            boxes: list in format [([up_left_x, up_left_y, bottom_right_x, bottom_right_y], conf),..]
                e.g. [(array([316.6288 , 186.26445, 374.53476, 413.9408 ], dtype=float32), 0.9993179),..]
            img: PIL image with drawn boxes
        '''
#         img_resized = cv2.resize(img, (self.size, self.size))
        img_resized = img.resize(size=(self.img_size, self.img_size))
        begin = time.time()
        detected_boxes = self.session.run(
            self.boxes, 
            feed_dict={self.inputs: [np.array(img_resized, dtype=np.float32)]}
        )
        suppressed_boxes = non_max_suppression(
            detected_boxes,
            confidence_threshold=self.conf_thresh,
            iou_threshold=self.iou_thresh
        ) 
        if verbose: print('PREDICTION time:', time.time() - begin)
        if draw_on_image:
            draw_boxes(
                suppressed_boxes, 
                img, 
                self.classes, 
                (self.img_size, self.img_size)
            )
        return suppressed_boxes, img
    
    
    def predict_tracker(
        self, 
        img, 
        frame_num=None, 
        verbose=False
    ):
        '''
        :param img: PIL.Image of shape (H, W, C) in colorspace in RGB format
        :param frame_num: number of the frame (for tracker)
        :param verbose: if True, print prediction time and ohter info
        
        :return predictions: list in format [(frame number, -1, center_x, center_y, box_width, box_height, probability, class_id, -1),..]
                e.g. [(array([24, -1, 374.53476, 413.9408, 100.5096, 400.7893, 0.9993179, 5, -1], dtype=float32),..]
        '''
#         img_resized = cv2.resize(img, (self.size, self.size))
        img_resized = img.resize(size=(self.size, self.size))

        begin = time.time()
        detected_boxes = self.session.run(
            self.boxes, 
            feed_dict={self.inputs: [np.array(img_resized, dtype=np.float32)]}
        )
        filtered_boxes = non_max_suppression(
            detected_boxes, 
            confidence_threshold=self.conf_thresh, 
            iou_threshold=self.iou_thresh
        )
        if verbose: print('PREDICTION time:', time.time() - begin)
        detection_size = (self.img_size, self.img_size)
        predictions = []
        for cls, bboxs in filtered_boxes.items():
            for box, score in bboxs:
                box = convert_to_original_size(
                    box, 
                    np.array(detection_size), 
                    np.array(img.size)
                )
                predictions.append(np.array([
                    frame_num, -1, 
                    box[0], box[1], box[2]-box[0], box[3]-box[1], 
                    score, cls, -1
                ]))
        return predictions
    
    
    def predict_batch(
        self, 
        batch, 
        verbose=False
    ):
        '''
        :param batch: batch of the np.array's (N, H, W, C) in RGB format
        :param verbose: if True, print prediction time and other info
        
        :return
            boxes: list of lists in format [
                ([up_left_x, up_left_y, bottom_right_x, bottom_right_y], conf),
                ...
                ] 
                each
            img: PIL image with drawn boxes
        '''

        begin = time.time()
        detected_boxes = self.session.run(
            self.boxes, 
            feed_dict={self.inputs: batch}
        )
        suppressed_boxes = non_max_suppression(
            detected_boxes, 
            confidence_threshold=self.conf_thresh, 
            iou_threshold=self.iou_thresh
        )
        if verbose: print('PREDICTION time:', time.time() - begin)
        return suppressed_boxes, batch
        

    def predict_multiple(
        self, 
        images: Tuple[np.ndarray],
        verbose=False,
        return_scores=False
    ) -> Dict[int, List[np.ndarray]]:
        '''
        Returns boxes of people for each input frame
        
        :param images: 
            views from different cameras at one moment
        :param return_scores:
            if True, returns a score (assurance) for each bounding box
            
        :return: 
            detections Dict[int, List[np.ndarray]], where key is cam_id and value is sequence of bboxes
        '''
        detections_dict = defaultdict(list)
        for cam_id, image in enumerate(images):
            img = Image.fromarray(image)
            img = img.resize(size=(self.img_size, self.img_size))
            begin = time.time()
            detected_boxes = self.session.run(
                self.boxes, 
                feed_dict={self.inputs: [np.array(img, dtype=np.float32)]}
            )
            suppressed_boxes = non_max_suppression(
                detected_boxes,
                confidence_threshold=self.conf_thresh,
                iou_threshold=self.iou_thresh
            )
            if verbose: print('PREDICTION time:', time.time() - begin)               
            suppressed_boxes = [
                detection for class_list in suppressed_boxes.values() 
                for detection in class_list
            ]  # all classes together
            for box in suppressed_boxes:
                box_orig = convert_to_original_size(
                    box[0],                         
                    np.array((self.img_size, self.img_size)),             
                    np.array((image.shape[1], image.shape[0]))
                )
                box_orig = np.array(box_orig, dtype='int')
                box_orig = np.clip(box_orig, 0, max(image.shape))
                detections_dict[cam_id].append(
                    (box_orig, box[1]) if return_scores else box_orig
                )
            if not len(suppressed_boxes):
                detections_dict[cam_id] = []       
        return detections_dict
    
    
    def predict_to_file(
        img_name, 
        pred_dir, 
        *args, 
        **kwargs
    ):
        '''
        :param img_name: name of image file to predict for
        
        :return: None, saves predictions to .xml-file with name == 'pred_dir/img_name'
        '''
        predictions = self.predict(*args, **kwargs)
        save_to_xml_format(img_name, e, pred_dir, predictions)
        