import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

from collections import defaultdict

from PIL import ImageDraw
import numpy as np

import cv2

from .utils import open_video, read_frames
from .types import Directory
from .registry import Registry  # old


class OldVisualizer(object):
    '''
    Class for demo.py results visualization (ids, boxes, timestamps..)    
    
    Example:
    visualizer = visualizers.Visualizer(registry_path='./test_registry.json')
    cam_ids = [0, 1]
    visualizer.visualize(cam_ids=cam_ids, input_dir=input_path, output_dir='./test_output')
    '''
    def __init__(self, registry_path: Directory, font=cv2.FONT_HERSHEY_SIMPLEX): 
        self.registry_path = registry_path
        '''
        History structure: {person_id: RegistryEntry}
        RegistryEntry: see :py:class:`.registry.RegistryEntry`
        '''
        self.registry = Registry.load(self.registry_path)
        '''Structure for the efficient (resource-reusable) visualization'''
        self.frame_history = None

        # draw params
        self.colors = {
            person_id: tuple(np.random.randint(low=0, high=255, size=3))
                for person_id in self.registry._entries
        }
        self.font = font

    def _draw_info(self, image, vis_info: Tuple):
        vis_image = image
        for person_id, bbox, till_visited in vis_info:
            color = self.colors[person_id] if till_visited else (255,0,0)  # else 'red'
            image = cv2.rectangle(
                image, 
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                0 if till_visited else 255, # COLOR == INT, REALLY??
                3,
            )
            image = cv2.putText(
                image,
                text=str(person_id),
                org=(bbox[0], bbox[1]),
                fontFace=self.font,
                fontScale=1,
                color=(255,255,255),
                thickness=2, 
                lineType=cv2.LINE_AA,
            )
        return image

    def _build_frame_history(self):
        '''
        Method to build self.frame_history only at the first visualization
        '''
        self.frame_history = defaultdict(lambda : defaultdict(list))

        for person_id, current_entry in self.registry._entries.items():
            for place in current_entry.places:
                self.frame_history[place['frame']][place['cam_id']].append(
                    (person_id, place['bbox'], current_entry.till_visited)
                )

    def visualize(self, cam_ids: List[int], input_dir: Directory, output_dir: Directory, verbose=False):
        '''
        Draws boxes and their person IDs given camera IDs 
        and other arguments (timestamps, particular events like "was not at cashier" etc.)

        :param cam_ids: numbers of cameras to visualize for
        '''
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        streams = sorted(input_dir.glob('*.mp4'))
        streams_filtered = [(cam_id, streams[cam_id].name) for cam_id in cam_ids]
        for cam_id, stream in streams_filtered:
            print(f'Camera number: {cam_id}, stream: {stream}')

        # efficient data structure
        if self.frame_history is None:
            self._build_frame_history()

        # create the output folder
        output_dir.mkdir(parents=True, exist_ok=True)

        # input -> ouput cycle
        # 191 shop TODO: config file
        fps = [20, 20, 20, 20, 22, 22, 23, 20, 15, 22]  # [13, 19]  # NEED TO GET FPS IN A PROPER WAY
        shapes = [(1280, 720), (1280, 720), (1280, 720),
                  (1280, 720), (1280, 960), (1280, 960),
                  (1280, 720), (1280, 720), (1280, 720),
                  (1280, 960)]
        if verbose:
            print('-------Visualization started-------')
        for (cam_id, stream_name), curr_fps, (width, height) in zip(streams_filtered, fps, shapes):
            if verbose:
                print(f'Camera id: {cam_id}, stream name: {stream_name}, fps: {curr_fps}, (w, h)={width, height}')
            with open_video(output_dir/stream_name, 'w', cv2.VideoWriter_fourcc(*'XVID'), curr_fps, (width, height)) as video_out:
                for frame_num, frame in enumerate(read_frames(input_dir/stream_name)):
                    vis_info = self.frame_history[frame_num][cam_id]  # [(person_id, bbox, till_visited), ..]
                    if verbose:
                        print(f'Frame {frame_num}, num people: {len(vis_info)}')
                    visualized_frame = self._draw_info(frame[:,:,::-1], vis_info)  # to BGR
                    video_out.write(visualized_frame)
