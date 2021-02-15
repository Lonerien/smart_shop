from collections import defaultdict

from typing import Dict, List, DefaultDict
import pickle
from pathlib import Path
import cv2
import os
import json
import jsonpickle
import csv
import time

from vkusmart.types import Timestamp, Filename, Places, Trajectory, ArmCrops, MulticamPosition, Position
# from vkusmart.types import GoodsPicks, ArmCoordHistory, ArmCropsHistory

class HistoryEntry(object):
    '''Represents information of one person's positions

    Usage is to set attributes as necessary
    In case of places you want to call .append() new place which is dict
    '''
    def __init__(
        self,
        track_id: int,
        store_enter: int,
        is_cassa_visited: bool=False,
        cassa_duration: int=0,
        cassa_enter: int=-1,
        cassa_exit: int=-1,
        store_exit: int=-1,
        pick_history = defaultdict(list),
        pick_after_cassa: int=0,
        trajectory: Trajectory=defaultdict(lambda: defaultdict(int))
    ):
        '''
        Args:
            начиная с cassa_duration заканчивая store_exit -- время совершения #действиеName#
            goods_in_basket: time ---> amount of goods were taken
            pick_history: time ---> amount of goods on cur frame
            pick_after_cassa -- взято после посещения кассы
            зная ID, получаем всю траекторию. Зная время, получаем коорд. со всех камер, с которых видим
            trajectory: dict{dict}  time ---> camera ---> BBox -- history of person locations
        '''
        self.track_id = track_id
        self.is_cassa_visited = is_cassa_visited
        self.cassa_duration = cassa_duration
        self.cassa_enter = cassa_enter
        self.cassa_exit = cassa_exit,
        self.store_enter = store_enter
        self.store_exit = store_exit
            
        self.grep_history = {} # track_id ---> № кадров, на которых взяли товар
        self.put_history = {} # track_id ---> № кадров, на которых взяли товар
        self.pick_after_cassa = 0
        self.trajectory = trajectory
        self.crops_count = 0

    def __str__(self):
        return (
            'RegistryEntry of:\n'
            f'person: {self.track_id}\n'
            f'bla-bla, copy from file documentation\n'
        )
            
    def add_in_hall(self,
                    frame_num,
                    cur_cam_idx,
                    is_cassa_visited,
                    new_position,
                    cas_in,
                    cas_out,
                   ):
        
        self.is_cassa_visited = is_cassa_visited # посещал ли кассу
        self.trajectory[frame_num][cur_cam_idx] = new_position #ведем историю перемещения
        if cas_in != -1: self.cassa_enter = cas_in # время подхода к кассовой зоне
        if cas_out != -1: self.cassa_exit = cas_out # время ухода из кассовой зоны
        
    def add_go_out_action(self,
                          frame_num: int,
                          is_paid: bool,
                          clusterized_grep: defaultdict(list),
                          clusterized_put: defaultdict(list)
                         ):
        self.store_exit = frame_num
        self.is_paid = is_paid
        self.grep_history = clusterized_grep
        self.put_history = clusterized_put
        
        self.pick_after_cassa = sum(grep_time > self.cas_out for grep_time in self.clusterized_grep) 
                         
        
        
class History(object):
    '''Sotres and provides interface for persons information

    TODO write __getitem__ instead of get method or inherit dict/defaultdict
    '''
    def __init__(self, history_write_dir: str, capacity=100):
        '''
        Entries maps person ids to their HistoryEntry
        '''
        self.history_write_dir = history_write_dir
        self._all_tracks_history: DefaultDict[int, HistoryEntry] = defaultdict(HistoryEntry)  
        self.capacity = capacity
        #self.fin_track: DefaultDict[int, bool] = defaultdict(HistoryEntry) # track_id --> bool (finished/not)
        self.fin_track = [] # list, хранящий ID ушедших из магазина людей. Когда его длина > capacity, дамп в scv
        self.start_time = str(time.time())

    def __str__(self):
        return f'History with {self._all_tracks_history.keys()} persons'
    
    def is_new(self, track_id: int):
        return not track_id in self._all_tracks_history

    def create(self, track_id: int, store_enter: int) -> None:
        '''Creates new track in History
        '''
        self._all_tracks_history[track_id] = HistoryEntry(track_id, store_enter)
        
    def history_sanitizer(self):
        ''' Dump info about finished tracks to the file
        '''
        for tr_id in self.fin_track: del self._all_tracks_history[track_id]
        self.fin_track.clear()
        
    def add(self, track_id: int, parametars):
        self._all_tracks_history[track_id].add(*parametars)
            
    def add_hall_info(self, track_id, person_info_from_hall):
        self._all_tracks_history[track_id].add_in_hall(*person_info_from_hall)
            
    def add_go_away_info(self, track_id, person_out_info):
        self._all_tracks_history[track_id].add_go_out_action(*person_out_info)
        self.fin_track.append(track_id)
        if len(self.fin_track) >= self.capacity:
            self.dump_to_csv(is_dump_all=False)
            self.history_sanitizer() # удаление треков вышедших людей
        #mb ++self.fin_track чет такое надо сделать
            
                                  
        

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
    
    def dump_to_csv(self, is_dump_all=False):
            
        csvData = [[
            'track_id', 'is_paid', 'is_cassa_visited', 
            'pick_count', 'greps_count', 'put_back_count'
        ]]
        
        for track_id, one_track in self._all_tracks_history.items():
            if not is_dump_all and track_id not in self.fin_track:
                continue
            csvData.append([
                track_id,
                self._all_tracks_history[track_id].is_paid,
                self._all_tracks_history[track_id].is_cassa_visited,
                len(self._all_tracks_history[track_id].grep_history),
                len(self._all_tracks_history[track_id].put_history)
            ])
        if not os.path.exists(self.history_write_dir):
            os.makedirs(self.history_write_dir)
        print("YOLO!")
        with open(self.history_write_dir+'/'+'history_report'+self.start_time+'.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(csvData)
            
