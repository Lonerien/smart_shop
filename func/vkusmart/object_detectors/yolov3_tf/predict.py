# -*- coding: utf-8 -*-

import os
from time import time

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

import cv2

from detector.detector import YOLOv3

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('names_file', '', 'File with class names')
tf.app.flags.DEFINE_string('weights_file', '', 'Binary file with detector weights')

tf.app.flags.DEFINE_string('input_path', '', 'Path to input image or input directory')
tf.app.flags.DEFINE_string('output_path', '', 'Path to output image or output directory')

tf.app.flags.DEFINE_string('video_path', '', 'Path to the video file to predict for')
tf.app.flags.DEFINE_integer('start_frame', 0, 'Frame where to start to detect')

tf.app.flags.DEFINE_integer('size', 608, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_integer('gpu_num', 0, 'Index of GPU-device to use')

# tf.app.flags.DEFINE_bool('as_yolo', True, 'If True, saves predictions to .txt file in YOLO format')


def load_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def write_info_yolo(fname, boxes, img, detection_size):
    '''
    Writes predicted bounding boxes to .txt file in YOLO-format 
    '''
    fname = fname[:fname.find('.')] + '.txt'
    frame_num = fname[:fname.find('.')]
    with open(fname, 'w+') as file:
        for cls, bboxs in boxes.items():
            for box, score in bboxs:
                box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
                # SCORE can be written too!!!
                file.write(f'{cls} {box[0]} {box[1]} {box[2]} {box[3]}' + '\n')

                
def predict_image(session, input_img, output_img, detector, save_preds=True):
    '''
    Predict bboxes for given image
    
    :param: session -- initialized tf.Session()
    :param: input_img -- path to input folder with images
    :param: output_img -- path to output folder (default='predictions')
    :param: detector -- detector.Detector() instance (for example, YOLOv3())
    :param: save_preds -- if True, saves images with drawn detections
    
    :return: None, saves all predictions in files
    '''
    
    begin = time()  # profile
    
    img = Image.open(input_img)

    boxes, img_with_boxes = detector.predict(img,
                                             iou_threshold=FLAGS.iou_threshold, 
                                             conf_threshold=FLAGS.conf_threshold)

    if save_preds:
        img_with_boxes.save(output_img)

    write_info_yolo(fname=output_img,
                    boxes=boxes,
                    img=img_with_boxes,
                    detection_size=(FLAGS.size, FLAGS.size))
        
    end = time()  # profile
    
    print(f'{output_img} predicted in {end - begin:.5f} seconds')
                

def predict_dir(session, input_dir, output_dir, detector):
    '''
    Predict bboxes for all images in given floder
    
    :param: session -- initialized tf.Session()
    :param: input_dir -- path to input folder with images
    :param: output_dir -- path to output folder (default='predictions')
    :param: detector -- detector.Detector() instance (for example, YOLOv3())
    
    :return: None, saves all predictions in files
    '''
    IMAGE_FORMATS = ['jpg', 'png', 'bmp']
    
    for img_name in os.listdir(input_dir):
        if img_name[-3:] not in IMAGE_FORMATS:
            continue
        predict_image(session, 
                      input_dir+os.sep+img_name, 
                      output_dir+os.sep+img_name, 
                      detector) 
        

def predict_video(session, input_path, output_path, detector, start_frame=0):
    '''
    Detect on video frame by frame with detector
    
    :param: session -- initialized tf.Session()
    :param: input_path -- path to input folder with images
    :param: detector -- detector.Detector() instance (for example, YOLOv3())
    :param: start_frame -- frame where to start to detect
    
    :return: None, shows all detections in another video
    '''
    VIDEO_FORMATS = ['mp4']
    if input_path[-3:] not in VIDEO_FORMATS:
        raise NotImplementedError()
    
    cap_in = cv2.VideoCapture(input_path)
    ret, frame = cap_in.read()
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     cap_out = cv2.VideoWriter(output_path,
#                               apiPreference=0,
#                               fourcc=fourcc,
#                               fps=24.0, 
#                               frameSize=frame.shape)

    i = -1
    while(cap_in.isOpened() and ret):
        i += 1
        if i < start_frame:
            continue

        rgb_pil_image = Image.fromarray(frame[...,::-1])
        
        _, img_with_boxes = detector.predict(rgb_pil_image,
                                             iou_threshold=FLAGS.iou_threshold,
                                             conf_threshold=FLAGS.conf_threshold,
                                             verbose=True)
        
#         out.write(np.array(img_with_boxes)[...,::-1])
        cv2.imwrite(output_path+os.sep+f'frame_{i}.jpg', np.array(img_with_boxes)[...,::-1])
        print(output_path+os.sep+f'frame_{i}.jpg')
        
        ret, frame = cap_in.read()
        
        
    cap_in.release()
#     cap_out.release()
               

def main(argv=None):

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # can be the other detector instance, for example, SSD()
    yolov3 = YOLOv3(session=sess, gpu_num=FLAGS.gpu_num, 
                    class_names=FLAGS.names_file,
                    weights_file=FLAGS.weights_file)

    if FLAGS.video_path != '':
        out_dir = FLAGS.output_path if FLAGS.output_path != '' else 'predictions'
        predict_video(sess, FLAGS.video_path, out_dir, yolov3, FLAGS.start_frame)
    else:    
        if os.path.isdir(FLAGS.input_path):
            out_dir = FLAGS.output_path if FLAGS.output_path != '' else 'predictions'
            predict_dir(sess, FLAGS.input_path, out_dir, yolov3)
        else:
            out_img = FLAGS.output_path if FLAGS.output_path != '' else 'predictions.jpg'
            predict_image(sess, FLAGS.input_path, out_img, yolov3)

        
if __name__ == '__main__':
    tf.app.run()
