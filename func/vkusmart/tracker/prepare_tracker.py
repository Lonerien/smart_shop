# import tensorflow as tf
from vkusmart.tracker.deep_sort import nn_matching
from vkusmart.tracker.deep_sort.tracker import Tracker
import argparse
from vkusmart.tracker.deep_sort.dataStorage import CameraInfo
from vkusmart.tracker.trackers_manager import TrackersManager

from vkusmart.types import CamsId

import os
import json
from collections import defaultdict


def prepare_tracker_release(config_file,
                            amount_camera,
                            desicion_maker_type,
                            exit_camera_idx,
                            pick_counter_path_to_neural_weights='',
                            draw_flag=True,
                            pickcounter_layout_dir='',
                            is_release=False):
    '''
    input:
    ----
    path to config file

    output:
    ----
    dsm_tracker:    the instance in dsm_tracker,
    Tuple:          the necessity of cold start (if True, then add video, camera id)
    '''
    if 'neural' in desicion_maker_type and pick_counter_path_to_neural_weights:
        print("##### THAT'S AN 'NEURAL' IN desicion_maker_type, but u didin't\
        specify any weights(pick_counter_path_to_neural_weights='')")
        
    cam_path = []
    best_cam_correlation_matrix = {}
    camera_types = defaultdict(str)

    camera_amount = 0
    cameras_id: CamsId = defaultdict(int)
    shelf_coords = defaultdict(list) # camera_idx ---> shelf_coords

    cam_area_match = CameraInfo()

    with open(config_file) as config_file:
        configs = json.load(config_file)
        for cur_config in configs:
            config = configs[cur_config]
            
            for one_camera in config['shelf_coords']:
                camera_idx = one_camera['camera_idx']
                shelf_coords[camera_idx] = one_camera['line']
        
            for camera_idx, camera_type in config['camera_types'].items():
                camera_types[int(camera_idx)] = camera_type
            
            for interseq_area in config['interseq_areas']:
                cam_area_match.add_new_interseq_area_from_json(interseq_area)

            for forgetfulness_area in config['forgetfulness_areas']:
                cam_area_match.add_new_area_forgetfulness_from_json(forgetfulness_area)

            for cassa_area in config['cassa_areas']:
                cam_area_match.add_new_cassa_area_from_json(cassa_area)

            for exit_area in config['exit_areas']:
                cam_area_match.add_new_exit_area_from_json(exit_area)

            cam_zones_amount = [9 for i in range(10)]

    metric = "cosine"
    print("prepare_tracker.py, release!")
    dsm_tracker = TrackersManager(
        metric,
        exit_camera_idx,
        out_dir='/home/user2/dsm_tracker/dsm_tracker/model_data/out',
        initial_camera=1,
        amount_camera=amount_camera,
        max_iou_distance=0.7,
        max_age=100,
        n_init=3,
        camera_types=camera_types,
        is_release=is_release,
        cam_area_match=cam_area_match,
        draw_flag=draw_flag,
        shelf_coords=shelf_coords,
        pickcounter_layout_dir=pickcounter_layout_dir,
        pick_counter_path_to_neural_weights=pick_counter_path_to_neural_weights,
        desicion_maker_type=desicion_maker_type
    )
    return dsm_tracker, (True, 1)
#     return dsm_tracker, (False, -1, -1)


if __name__ == '__main__':
    prepare_tracker_release('test_config')
