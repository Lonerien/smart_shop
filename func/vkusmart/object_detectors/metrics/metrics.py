import os
import glob
from pathlib import Path
import argparse

from collections import Counter

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import skimage.io


def load_class_names(names_file):
    '''
    Return id_to_class and class_to_id mappings
    '''
    id_to_class = {}
    class_to_id = {}
    with open(names_file, 'r+') as file:
        for i, line in enumerate(file):
            id_to_class[i] = line.strip()
            class_to_id[line.strip()] = i
    return id_to_class, class_to_id


def load_info_xml(filename, CLASS_TO_ID, fmt='voc'):
    '''
    Load bounding boxes with class_id, coords and confidence from .xml file
    
    :args:
        filename -- name of file with info about bboxes
        fmt -- format of bboxes: 'voc' -- read PASCAL VOC-like .xml file with bounding boxes
    :return:
        info -- dict with keys: 'class_id', 'coords', 'conf' (None if ground truth) 
    '''
    e = ET.parse(filename).getroot()
    bboxes = []
    for obj in e.findall('object'):
        class_id = CLASS_TO_ID[obj.find('name').text]
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        conf = bbox.find('conf')
        if conf is not None:
            conf = float(conf.text)
        if fmt == 'voc':
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            bboxes.append({'class_id': class_id, 
                           'coords': (x1, y1, x2, y2), 
                           'conf': conf})
        elif fmt == 'coco':
            w = float(bbox.find('w').text)
            h = float(bbox.find('h').text)
            bboxes.append({'class_id': class_id, 
                           'coords': (x1, y1, w, h), 
                           'conf': conf})
    return bboxes


def load_info_yolo(filename, img_size):
    '''
    Load bounding boxes with class_id, coords and confidence from .txt file in YOLO-format
    
    :args:
        filename -- name of file with info about bboxes
        img_size -- original image (height, width)
    :return:
        info -- dict with keys: 'class_id', 'coords', 'conf' (None if ground truth) 
    '''
    bboxes = []
    img_height = img_size[0]
    img_width = img_size[1]
    with open(filename, 'r+') as file:
        for line in file:
            class_id, center_x, center_y, width, height = [float(x) for x in line.split(' ')]
            # to original size
            center_x *= img_width
            center_y *= img_height
            width *= img_width
            height *= img_height
            # to (x1, y1, x2, y2) corners format
            w2 = width / 2
            h2 = height / 2
            x1 = center_x - w2
            y1 = center_y - h2
            x2 = center_x + w2
            y2 = center_y + h2
            # confidence can be added later
            bboxes.append({'class_id': int(class_id), 
                           'coords': (x1, y1, x2, y2), 
                           'conf': None})
    return bboxes


def load_info_yolotf(filename):
    '''
    Load bounding boxes with class_id, coords and confidence from .txt file in YOLO-TensorFlow-format
    
    :args:
        filename -- name of file with info about bboxes
    :return:
        info -- dict with keys: 'class_id', 'coords', 'conf' (None if ground truth) 
    '''
    bboxes = []
    with open(filename, 'r+') as file:
        for line in file:
            class_id, x1, y1, x2, y2 = [float(x) for x in line.split(' ')]
            # confidence can be added later
            bboxes.append({'class_id': int(class_id), 
                           'coords': (x1, y1, x2, y2), 
                           'conf': None})
    return bboxes


def iou(boxA, boxB):
    '''
    Calculate the Intersection over Union of two bounding boxes.
    Boxes must be represented as (x1, y1, x2, y2): upper left and lower right corners
    '''
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def metrics_per_class(gt_dir, pred_dir, fmt, names_file, 
                      iou_thresh=0.5, conf_thresh=0.5, save_to_file=True):
    '''
    Calculate metrics (precision and recall) values for each class
    
    :args:
        classes -- dict with IDs of all classes as keys 
                   and names of classes as values (e.g. {0: 'butter', 1:'bread', ..})
        gt_dir -- folder with ground truth bounding boxes files
        pred_dir -- folder with predicted bounding boxes files
        iou_thresh -- threshold of IoU for predicted and GT bboxex overlap
        conf_thresh -- confidence threshold strting from which bbox is taken into account
        save_to_file -- if True, save the result table to a file
    :return:
        metrics_df -- pd.DataFrame with class_names in columns
        
    NOTE: number of files with bboxes must be the same in the gt_dir and in the pred_dir, 
    each ground truth file must match the name of a predicted file
    '''
    
    ID_TO_CLASS, CLASS_TO_ID = load_class_names(names_file)
    NUM_CLASSES = len(list(ID_TO_CLASS.items()))
    
    class_all = Counter()  # per class counts along all test sample
    class_tp = Counter()  # per class True Positives
    class_fp = Counter()  # per class False Positives
    
    overlayed_obj_images = []
    false_positives = []
    
    collision_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    print(collision_matrix)

    # for every single image calculate precision and recall
    for gt_file, pred_file in zip(glob.glob(gt_dir+os.sep+f'*.{fmt}'), 
                                  glob.glob(pred_dir+os.sep+f'*.txt')):
        if fmt == 'txt':
            img_size = skimage.io.imread(gt_file[:-4] + '.jpg').shape
            gt_info = load_info_yolo(gt_file, img_size)
        elif fmt == 'xml':
            gt_info = load_info_xml(gt_file, CLASS_TO_ID)
        else:
            raise ValueError('Format ' + fmt + ' is not supported')
        pred_info = load_info_yolotf(pred_file)
        print(gt_file, pred_file)
        pred_info = list(filter(lambda bbox: (bbox['conf'] is None) 
                                 or (bbox['conf'] >= conf_thresh), pred_info))

        iou_matrix = np.array([np.array([iou(bbox1['coords'], bbox2['coords']) 
                                          for bbox2 in pred_info]) for bbox1 in gt_info])
#         print(iou_matrix)
        iou_matrix = iou_matrix >= iou_thresh
        # can be done with broadcasting, this variant is for readability
        class_match_matrix = np.array([np.array([gt_info[i]['class_id'] == pred_info[j]['class_id'] 
                                                  for j in range(len(pred_info))]) 
                                                   for i in range(len(gt_info))])
        class_not_match_matrix = 1 - class_match_matrix
        # rows == ground truth boxes
        # columns == predicted boxes
        # IoU > than iou_thresh and classes are the same
        detect_matrix = iou_matrix * class_match_matrix
        detect_collisions_matrix = iou_matrix * class_not_match_matrix
        
        # check collisions of goods with each other 
        # (sometimes it is just high IoU with another class, no mistake)
        collisions = np.where(detect_collisions_matrix)
        for i, j in zip(collisions[0], collisions[1]):
            if detect_matrix[:, j].sum() == 0:  # bbox isn't true positive
                collision_matrix[gt_info[i]['class_id']][pred_info[j]['class_id']] += 1

        gt_detections = detect_matrix.sum(axis=1)
        pred_detections = detect_matrix.sum(axis=0)
        for i, det in enumerate(gt_detections):
            class_all[gt_info[i]['class_id']] += 1  # TP + FN
            if det > 0:
                class_tp[gt_info[i]['class_id']] += 1  # TP
        for j, det in enumerate(pred_detections):
            if det == 0:
                class_fp[pred_info[j]['class_id']] += 1  # FP -- pred_box without gt_box
                false_positives.append(pred_file)
            elif det > 1:
                overlayed_obj_images.append(pred_file)
#         print('-------------------------------------------')

    recall = {class_id : round(class_tp[class_id] / class_all[class_id], 3) 
              for class_id in class_all.keys()}
    precision = {class_id : round(class_tp[class_id] / (class_tp[class_id] + class_fp[class_id]), 3) 
                 for class_id in class_all.keys()}
    
    # metrics DataFrame
    metrics_df = pd.DataFrame(data=[recall, precision])
    metrics_df.columns = [ID_TO_CLASS[col] for col in metrics_df.columns]
    metrics_df.index = ['recall', 'precision']
    metrics_df = metrics_df.T
    if save_to_file:
        metrics_df.to_excel('./metrics.xls')
        
    # collisions DataFrame
    collisions_df = pd.DataFrame(data=collision_matrix)
    collisions_df.columns = [ID_TO_CLASS[col] for col in collisions_df.columns]
    collisions_df.index = [ID_TO_CLASS[idx] for idx in collisions_df.index]
    if save_to_file:
        collisions_df.to_excel('./collisions.xls')
    
    print('Number of images with strongly overlayed objects:', len(overlayed_obj_images))
    print('Number of images with false positives:', len(false_positives))

    with open('overlayed_obj_images.txt', 'w+') as over_file:
        for name in overlayed_obj_images:
            over_file.write(name[:-3]+'jpg'+"\n")
            over_file.write(name[:-3]+'txt'+"\n")
            
    with open('false_positives.txt', 'w+') as fp_file:
        for name in false_positives:
            fp_file.write(name[:-3]+'jpg'+"\n")
            fp_file.write(name[:-3]+'txt'+"\n")

    return metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gt_dir', 
                        help='folder with ground truth .txt files with bounding boxes', required=True)
    parser.add_argument('-p', '--pred_dir', 
                        help='folder with predicted .txt files with bounding boxes', required=True)
    parser.add_argument('-n', '--names_file', 
                        help='.names file with names of classes', required=True)
#     parser.add_argument('-y', '--yolo_format', 
#                         help='if True, assume that bounding boxes are in the original YOLO format \
#                         (float numbers -- relative coordinates)', default=0, required=False)
    parser.add_argument('arg', nargs='*') # use '+' for 1 or more args (instead of 0 or more)
    args = parser.parse_args()
    
    metrics_df = metrics_per_class(gt_dir=args.gt_dir, pred_dir=args.pred_dir, fmt='xml', 
                                   names_file=args.names_file)

if __name__ == "__main__":
    main()
