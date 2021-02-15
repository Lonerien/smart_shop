# ------------------------------------------------------------------------------
# Copyright (c) MIPT NeurusLab, 2019
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path

import numpy as np
from easydict import EasyDict as edict


config = edict()

# --- Directories ---
config.DIRS = edict()
config.DIRS.ROOT_DIR = Path('/mnt/nfs/vkusmart/master/').resolve()
config.DIRS.DATA_DIR = config.DIRS.ROOT_DIR/'data'
config.DIRS.SESSION_DIR = config.DIRS.DATA_DIR/'session_test'
config.DIRS.VIDEOS_DIR = config.DIRS.SESSION_DIR/'input_videos'
config.DIRS.PICKCOUNTER_LAYOUT_DIR = config.DIRS.DATA_DIR/'test_pick_counter/seq1/right_cam/layouts'
config.DIRS.PERSONS_DIR = config.DIRS.SESSION_DIR/'persons'
config.DIRS.OUTPUT_VIDEOS_DIR = config.DIRS.SESSION_DIR/'output_videos'
config.DIRS.ROGUES_VIDEOS_DIR = config.DIRS.OUTPUT_VIDEOS_DIR/'rogues'
config.DIRS.DEBUG_VIS_DIR = config.DIRS.OUTPUT_VIDEOS_DIR/'debug_vis'
config.DIRS.WEIGHTS_ROOT = Path('/mnt/nfs/vkusmart/master/vkusmart/weights/').resolve()

# --- Input videos params ---
config.INPUT = edict()
config.INPUT.NUM_VIDEOS = 1
config.INPUT.HEIGHT = 1080
config.INPUT.WIDTH = 1920
config.INPUT.FPS = 20

# --- cuDNN related params ---
config.CUDNN = edict()
config.CUDNN.BENCHMARK = False
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# --- Detector ---
config.DETECTOR = edict()
config.DETECTOR.FRAMEWORK = 'tf'  # ['tf', 'pytorch']
config.DETECTOR.ARCH = 'yolov3'
config.DETECTOR.DATASET = 'persons-vkusvill'  # taken from Neurus team
config.DETECTOR.NUM_CLASSES = 1  # class "person"
config.DETECTOR.CONFIG_PATH = config.DIRS.WEIGHTS_ROOT/'detectors/yolov3-persons.cfg'
config.DETECTOR.WEIGHTS_PATH = config.DIRS.WEIGHTS_ROOT/'detectors/yolov3-persons_final0609.weights'
config.DETECTOR.NAMES_PATH = config.DIRS.WEIGHTS_ROOT/'detectors/persons.names'
config.DETECTOR.IMG_SIZE = 640
config.DETECTOR.CONF_THRESH = 0.5
config.DETECTOR.IOU_THRESH = 0.4
config.DETECTOR.GPU_NUM = 0  # [0, 1, 2]

# --- Re-identification ---
# config.REID = edict()
# config.REID.FRAMEWORK = 'torchreid'  # ['torchreid', 'open-reid']
# config.REID.ARCH = 'resnet50-mid'  # ['resnet50-mid']
# config.REID.LOSS = 'softmax'  # ['softmax', 'triplet']
# config.REID.FEATURE_LAYER = 'special_linear_bn'
# config.REID.DATASET = 'vkusmart_v2_am'
# config.REID.WEIGHTS_PATH = config.DIRS.WEIGHTS_ROOT/'reids/torchreid/resnet50mid-softmax-vkusmart_v2-staged_lr_adam003/model.pth.tar-60'
# config.REID.GPU_NUM = 0  # [0, 1, 2]
config.REID = edict()
config.REID.FRAMEWORK = 'torchreid'  # ['torchreid', 'open-reid']
config.REID.ARCH = 'osnet_ibn_x1_0'  # ['resnet50-mid']
config.REID.LOSS = 'softmax'  # ['softmax', 'triplet']
#config.REID.FEATURE_LAYER = 'special_linear_bn'
config.REID.DATASET = 'vkusmart_v4_am'

# log/osnet_x1_0-softmax-vkusmart_v5_am__random_erase_user5/model.pth.tar-50
# config.REID.WEIGHTS_PATH = config.DIRS.WEIGHTS_ROOT/'reids/torchreid/osnet_ibn_x1_0-softmax-vkusmart_v4_am_user5/model.pth.tar-60'
config.REID.WEIGHTS_PATH = Path('/mnt/nfs/vkusmart/reid/deep-person-reid/log/osnet_x1_0-softmax-vkusmart_v5_am__random_erase_user5/model.pth.tar-50').resolve()  # с ними демка от 10.10
# config.REID.WEIGHTS_PATH = Path('/mnt/nfs/vkusmart/reid/deep-person-reid/log/osnet_x1_0-softmax-vkusmart_v5_am__random_erase_user5_1/model.pth.tar-80').resolve() # с ними самая первая демка



config.REID.GPU_NUM = 0  # [0, 1, 2]

# --- Tracker ---
config.TRACKER = edict()
config.TRACKER.CONFIG_PATH = ''

# --- History ---
config.HISTORY = edict()
config.HISTORY.SAVING_DIR = ''

# --- Pose Estimator ---
config.POSEMODEL = edict()
config.POSEMODEL.FRAMEWORK = 'AlphaPose'
config.POSEMODEL.DATASET = 'coco2017'
config.POSEMODEL.NUM_JOINTS = 17
config.POSEMODEL.CROP_HEIGHT = 256  # [256, 320]
config.POSEMODEL.CROP_WIDTH = 192  # [192, 256]
config.POSEMODEL.HM_HEIGHT = 64  # [64, 80]
config.POSEMODEL.HM_WIDTH = 48  # [64, 48]
config.POSEMODEL.FAST_INFERENCE = False  # [False, True]
config.POSEMODEL.HM_MODEL_NAME = 'pose_resnet50'  # ['hourglass', 'pose_resnet50']
config.POSEMODEL.HM_WEIGHTS_PATH = config.DIRS.WEIGHTS_ROOT/'posemodels/pose_resnet_50_256x192.pth.tar'
config.POSEMODEL.GPU_NUM = 0  # [0, 1, 2]
# simple baselines legacy code for PoseResnet50
POSE_RESNET = edict()
POSE_RESNET.NUM_LAYERS = 50
POSE_RESNET.DECONV_WITH_BIAS = False
POSE_RESNET.NUM_DECONV_LAYERS = 3
POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
POSE_RESNET.FINAL_CONV_KERNEL = 1
POSE_RESNET.TARGET_TYPE = 'gaussian'
POSE_RESNET.HEATMAP_SIZE = [
    config.POSEMODEL.HM_WIDTH, 
    config.POSEMODEL.HM_HEIGHT
]
POSE_RESNET.SIGMA = 2
# only for simple-baselines repo compatibility
config.MODEL = edict()
config.MODEL.EXTRA = POSE_RESNET

# --- Armstate classifier ---
config.ARMSTATE = edict()
config.ARMSTATE.ARCH='mobilenet_v2' #'efficientnet-b0'  # ['resnet18', 'resnet50', 'efficientnet-b0']
config.ARMSTATE.NUM_CLASSES = 4 # [3, 4]  # 0="empty", 1="with an item", 2="trash", 3="bag/pocket"
config.ARMSTATE.CROP_SIZE = 150 #125

# config.ARMSTATE.WEIGHTS_PATH = config.DIRS.WEIGHTS_ROOT/'armstate/armstate_v7_efficientnet-b0_unfrozen_comlicated_ptd_0.96.pth'
config.ARMSTATE.WEIGHTS_PATH = Path('/mnt/nfs/user1/models/armstate_v8_comb_rot_mobilenet_v2_unfrozen_complicated_ptd_0.921.pth').resolve()

config.ARMSTATE.GPU_NUM = 0  # [0, 1, 2]

## resnet18 (ks): 'armstate/armstate_v6_resnet18_0.891.pth'
## resnet50 (ai): 'armstate/armstate_v7_resnet50_0.969.pth'
## efficientnet-b0 (ai): 'armstate/armstate_v7_efficientnet-b0_unfrozen_comlicated_ptd_0.96.pth'
## mobilenet-v2 (ai): 'armstate/armstate_v7_mobilenet_v2_unfrozen_simplified_ptd_0.969.pth'

# !!! NOT USED YET, HERE FOR REMINDER: !!!
config.TEST = edict()
# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True
config.TEST.USE_GT_BBOX = False
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = '/mnt/nfs/pose/human-pose-estimation.pytorch/models/pytorch/pose_coco/pose_resnet_50_256x192.pth.tar'
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


# def update_dir(model_dir, log_dir, data_dir):
#     if model_dir:
#         config.OUTPUT_DIR = model_dir

#     if log_dir:
#         config.LOG_DIR = log_dir

#     if data_dir:
#         config.DATA_DIR = data_dir

#     config.DATASET.ROOT = os.path.join(
#             config.DATA_DIR, config.DATASET.ROOT)

#     config.TEST.COCO_BBOX_FILE = os.path.join(
#             config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

#     config.MODEL.PRETRAINED = os.path.join(
#             config.DATA_DIR, config.MODEL.PRETRAINED)


# def get_model_name(cfg):
#     name = cfg.MODEL.NAME
#     full_name = cfg.MODEL.NAME
#     extra = cfg.MODEL.EXTRA
#     if name in ['pose_resnet']:
#         name = '{model}_{num_layers}'.format(
#             model=name,
#             num_layers=extra.NUM_LAYERS)
#         deconv_suffix = ''.join(
#             'd{}'.format(num_filters)
#             for num_filters in extra.NUM_DECONV_FILTERS)
#         full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
#             height=cfg.MODEL.IMAGE_SIZE[1],
#             width=cfg.MODEL.IMAGE_SIZE[0],
#             name=name,
#             deconv_suffix=deconv_suffix)
#     else:
#         raise ValueError('Unkown model: {}'.format(cfg.MODEL))

#     return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
