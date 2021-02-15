from collections import defaultdict

from typing import Dict, List, DefaultDict
import pickle
from pathlib import Path
import cv2
import os
import json
import jsonpickle

from vkusmart.types import Timestamp, Filename, Places, GoodsPicks, ArmCoordHistory, ArmCropsHistory, Trajectory, ArmCrops, MulticamPosition, Position


class HistoryEntry(object):
    '''Represents information of one person's positions

    Usage is to set attributes as necessary
    In case of places you want to call .append() new place which is dict
    '''
    def __init__(
        self,
        track_id: int,
        is_cassa_visited: bool=False,
        cassa_duration: int=0,
        cassa_enter: int=-1,
        cassa_exit: int=-1,
        store_enter: int=-1,
        store_exit: int=-1,
        pick_count: int=0,
        pick_history: GoodsPicks=[],
        pick_after_cassa: int=0,
        arm_crops: ArmCropsHistory=[],
        arm_coords: ArmCoordHistory=[],
        trajectory: Trajectory=defaultdict(lambda: defaultdict(list)),
    ):
        '''
        Args:
            track_id: -----------------------------
            places: list of person's places, see :py:class:`.types.Place`
            is_cassa_visited: ---------------------
            time_till_enter: time of enterance to tills zone
            time_till_exit: time of exit from tills zone
            pick_count: amount od goods were taken
            pick_history: list of pairs (time, camera) -- add new, when person take good
            
            pick_after_cassa: ---------------------
            зная ID , получаем всю историю. Зная время, получаем кропы и КП со всех камер, с которых видим
            arm_coords: dict[list]  time --> list(camera, BBox_KeyPoints ) need for testing armstate
            
            arm_crops: кропы-изображения
            
            зная ID, получаем всю траекторию. Зная время, получаем коорд. со всех камер, с которых видим
            trajectory: list[dict]  time --> dict{camera, BBox} -- history of person locations
        '''
        self.track_id = track_id
        self.is_cassa_visited = is_cassa_visited
        self.cassa_duration = cassa_duration
        self.pick_count = pick_count
        self.pick_history = pick_history
        self.pick_after_cassa = pick_after_cassa
        self.arm_crops = arm_crops
        self.arm_coords = arm_coords
        self.trajectory = trajectory
        self.cassa_enter = cassa_enter
        self.cassa_exit = cassa_exit
        self.store_enter = store_enter
        self.store_exit = store_exit
        
        self.crops_count = 0

    def __str__(self):
        return (
            'RegistryEntry of:\n'
            f'person: {self.track_id}\n'
            f'bla-bla, copy from file documentation\n'
        )
    
    def add(
        self,
        time: int,
        is_cassa_visited: bool,
        cassa_duration: int,
        pick_count: int,
        cur_frame_goods_picks: GoodsPicks,
        pick_after_cassa: int,
        new_arm_crops: ArmCrops,
        new_arm_coords: ArmCoordHistory,
        new_position: Position,
        cassa_enter: int=-1,
        cassa_exit: int=-1,
        store_enter: int=-1,
        store_exit: int=-1,
    ):
        self.is_cassa_visited = is_cassa_visited
        self.cassa_duration = cassa_duration
        self.pick_count = pick_count
        self.pick_history.append(cur_frame_goods_picks)
        self.pick_after_cassa = pick_after_cassa
        self.arm_crops.append(new_arm_crops)  # или тоже [time]
#         self.arm_coords[time] = new_arm_coords
        self.trajectory[time][new_position[0]] = new_position[1] # [0] -- camera, [1] -- bbox
        if cassa_enter != -1: self.cassa_enter = cassa_enter
        if cassa_exit != -1: self.cassa_exit = cassa_exit
        if store_enter != -1: self.store_enter = store_enter
        if store_exit != -1: self.store_exit = store_exit
        
        
class History(object):
    '''Sotres and provides interface for persons information

    TODO write __getitem__ instead of get method or inherit dict/defaultdict
    '''
    def __init__(self, arm_crops_dir=None, capacity=100):
        '''
        Entries maps person ids to their HistoryEntry
        '''
        self._all_tracks_history: DefaultDict[int, HistoryEntry] = defaultdict(HistoryEntry)  
        self.arm_crops_dir = arm_crops_dir  
        self.crops_count = 0 
        self.capacity = capacity
        self.fin_track: DefaultDict[int, bool] = defaultdict(HistoryEntry) # track_id --> bool (finished/not)
        self.disc_chank = 0

    def __str__(self):
        return f'History with {self._all_tracks_history.keys()} persons'
    
    def is_new(self, track_id: int):
        return not track_id in self._all_tracks_history

    def create(self, track_id: int) -> None:
        '''Creates new track in History
        '''
        self._all_tracks_history[track_id] = HistoryEntry(track_id)
        
    def history_sanitizer(self):
        ''' Dump info about finished tracks to the file
        '''
        print('###########################call history_sanitizer')
        self.dump_to_disk(mode='write_finished', track_boxes=True)
        self.disc_chank += 1
        self.fin_track.clear()
        
    def add(self, track_id: int, parametars):
        self._all_tracks_history[track_id].add(*parametars)
        if len(self.fin_track.keys()) >= self.capacity:
            self.history_sanitizer()
        

    def get(self, track_id: int):
        '''Returns history of the track by ID

        This method is used for both reading and updating Entries
        '''
        return self._all_tracks_history[track_id]

    def dump(self, filename: Filename, *, method: str='.pickle') -> None:
        '''Saves current registry state to given 
        '''
        filename = Path(filename).with_suffix(method)
        with open(filename, 'wb') as file:
            pickle.dump(self._all_tracks_history, file)

    @classmethod
    def load(cls, filename: Filename):
        registry = cls()
        with open(filename, 'rb') as file:
            entries = pickle.load(file)
        registry._all_tracks_history = entries
        return registry
    
    def toJSON(self):
        print('##############json dump -- some discussion needed##############')
        with open('json_dump_test2.json', 'a') as outfile:
            for track_id, one_track in self._all_tracks_history.items():
                print('here ', track_id, one_track.is_cassa_visited, one_track.pick_count, one_track.pick_after_cassa)
                json_string = """
                {
                    "track": {
                        "track_id": "%d",
                        "is_cassa_visited": "%s",
                        "pick_count": "%d",
                        "pick_after_cassa": "%d",
                    }
                }
                """ % (track_id, one_track.is_cassa_visited, one_track.pick_count, -1)
                json.dump(json_string, outfile)
            

    def dump_to_disk(self, mode='regular',is_arm_crops=False, keypoints=False, track_boxes=False, track_info=False):
        '''Method that saves needed information right to the disk in specified formats.
            E.g. if arm_crops=True, then it will save all the new crops that are got on last iteration
            of the pipeline to the spicified folders on disk
        '''
        tracks_history: DefaultDict[int, HistoryEntry] = defaultdict(HistoryEntry) 
            
        if mode=='regular':
            tracks_history = self._all_tracks_history
        if mode=='write_finished':
            for track_id, track_info in self._all_tracks_history.items():
                if track_id in fin_track.keys():
                    tracks_history[track_id] = track_info
        
        if is_arm_crops:
            for track_id in self._all_tracks_history:
                one_person_info = self._all_tracks_history[track_id]
                this_person_folder = self.arm_crops_dir+'/'+str(track_id)+'/'
                
                if not os.path.exists(this_person_folder):
                    os.makedirs(this_person_folder)
                
                #cv2.imwrite(this_person_folder+str(self.crops_count)+'.jpg', one_person_info['crops_info']['crops'])
                # wait... smth wrong...
                
                for arm_crop_at_cameras in one_person_info.arm_crops:
#                     for camera_idx in arm_crop_at_cameras.keys():
                    for arm_type in arm_crop_at_cameras:
#                         print('type(arm_crop_at_cameras[arm_type]=', type(arm_crop_at_cameras[arm_type]))
                        if type(arm_crop_at_cameras[arm_type]) == list: # list brake everything
                            break
                        cv2.imwrite(this_person_folder+str(self.crops_count)+'.jpg', arm_crop_at_cameras[arm_type])
                        self.crops_count += 1
        if track_boxes:
            for track_id, one_person_info in tracks_history.items():
                
                this_person_folder = self.arm_crops_dir+'/persons/'+str(track_id)+'/'
                
                if not os.path.exists(this_person_folder):
                    os.makedirs(this_person_folder)
                
                js = jsonpickle.encode(history[track_id])
                with open(this_person_folder + 'info.json', 'w+') as outfile:
                    json.dump(js, outfile)
                
#                 for time, multicam_bbox in one_person_info.trajectory.items():
#                     for camera, bbox in multicam_bbox.items():
#                         print('camera=', camera)
#                         print('bbox=', bbox)
                        # по итого, это записываем в json или в бинарник?
        
        if track_info:
            self.toJSON()
            
                    
                
