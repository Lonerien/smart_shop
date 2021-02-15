from typing import Iterable, Tuple
from pathlib import Path
from itertools import count

import numpy as np

from .types import Images, Timestamp, Directory
from .utils import read_frames, return_path


class VideoProvider(object):
    '''Interface for data acess
    '''
    def frames(self) -> Iterable[Tuple[int, Timestamp, Images]]:
        '''Generator of simultaneous frames from devices

        Yields:
            (frame_number, timestamp, images)
        '''
        pass


class OfflineVideoProvider(VideoProvider):
    '''Provides frames from videos on disk
    '''
    def __init__(self, data_dir: Directory, wildcard: str='*.mp4', start_frame_numbers: list=[]):
        '''
        Args:
            data_dir: str of Path to directory with videos to stream
            wildcard: pattern to filter videos, see :py:funct:`pathlib.Path.glob`
        '''
        self.data_dir = Path(data_dir)
        self.wildcard = wildcard
        self.videos_paths = sorted(self.data_dir.glob(self.wildcard))
        for i,video_path in enumerate(self.videos_paths):
            print(i, '<----->', video_path) 
        if start_frame_numbers == []:
            print('no "start_frame_numbers" shifts in videos')
            self.start_frame_numbers = [0 for i in range(len(self.videos_paths))]
        else:
            self.start_frame_numbers = start_frame_numbers

    def frames(self) -> Iterable[Tuple[int, Timestamp, Images]]:
        '''Iteratively provides frames from multiple videos situated on disk

        Note: yields frame number as timestamp
        '''
        yield from zip(count(), count(), zip(*[read_frames(path, s_f_n) for path, s_f_n in zip(self.videos_paths, self.start_frame_numbers)]))
        
#     def frames_with_name(self) -> Iterable[Tuple[int, str, Images]]:
#         '''Iteratively provides frames from multiple videos situated on disk

#         Note: yields frame number as timestamp
#         '''
#         yield from zip(count(), zip(*[return_path(path) for path in self.videos_paths]), zip(*[read_frames(path) for path in self.videos_paths]))
        
    
        
    def show_frames_in_one_stream(self, path_to_video) -> Iterable[Tuple[int, Timestamp, Images]]:
        yield from zip(count(), count(), zip(*[read_frames(path) for path in self.videos_paths if path==path_to_video]))
        
        
class OneStreamVideoProvider(VideoProvider):
    '''Provides frames from videos on disk
    '''
    def __init__(self, video_name: str):
        '''
        Args:
            video_name: str == full path to the video file
        '''
        self.video_name = video_name

    def frames(self) -> Iterable[Tuple[int, Timestamp, Images]]:
        '''Iteratively provides frames from multiple videos situated on disk

        Note: yields frame number as timestamp
        '''
        yield from zip(count(), count(), zip(*[read_frames(self.video_name)]))