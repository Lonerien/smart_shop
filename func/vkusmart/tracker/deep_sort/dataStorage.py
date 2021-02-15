from collections import defaultdict
from . import nn_matching 
import numpy as np
import time

        
class Camera_interseq_areas(object):
    '''
    Interseq_areas -- areas, that can be seen from two or more cameras. 
    
    We have area, seen from some camera(yes, we don't know it's ID).
 
    self.otherCamIds_areas -- cam_ids, that see that area too
    otherCamIds_areas: dict of lists
            other_cam_id ------ id of other camera, that see that area too(now, its unnessesary field)
            area -------------- bounding box of area
            area_match_index -- area_index. We need it, cause 2 cameras, can see 2 areas. 
                                Areas matches, using area_match_index
            (for example, a long hall, that splited into 2 parts by the shelf,
            and we want to differ people in the right side from people in the left side)
            
    self.persons_inside_area: defaultdict intersection_cam_id -> list
        where list contain: track_id, track_features, zone_idx
        
        after every Tracker.update() we have the most recent list of persons, who has tracks,
        and who located in area 
        
        to get person info, you need to know zone_idx
            
    '''
    def __init__(self):                         # поле area_match_index лишнее, но оставленно для совместимости (ее, legacy)
        self.otherCamIds_areas = {}   # area_match_index -> other_cam_id, area, area_match_index 
        self.persons_inside_area = defaultdict(list)  
    
    def add_new(self, otherCamId_area):        # Один индекс ---> одна зона
        self.otherCamIds_areas[otherCamId_area[2]] = otherCamId_area   # other_cam_id, area, area_match_index
        
    def bbox_belongs_to_zone(self, xy1xy2, eps=0):
        all_zones = []
        for area_idx, otherCam_area in self.otherCamIds_areas.items():
#             print('otherCam_area=', otherCam_area)
            zone = otherCam_area[1]
            if xy1xy2[0] > zone[0]-eps and xy1xy2[1] > zone[1]-eps and xy1xy2[2] < zone[2]+eps and xy1xy2[3] < zone[3]+eps:
                all_zones.append((True, otherCam_area[0], otherCam_area[2]))
        if all_zones == []:
            return [(False, -1, -1)]
        return all_zones
#         return False, -1, -1 # otherCam_area[2]
    
    def add_update_person_inside_area(self, track_id, otherCamId_area, track_features, zone_idx):
        self.persons_inside_area[otherCamId_area].append((track_id, track_features, zone_idx))
    
    def get_all_persons_inside_area(self, otherCamId_area, zone_idx):
        persons = []
        for pers in self.persons_inside_area[otherCamId_area]:
            if pers[2] == zone_idx:
                persons.append(pers)
        return persons
    
    def get_priority(self, area_index):
#         for area in self.otherCamIds_areas:
#             print(area)
#             print('----------')
#         print('area_index=', area_index)
#         print(self.otherCamIds_areas[area_index])
#         print(self.otherCamIds_areas[area_index][0])
#         print(self.otherCamIds_areas[area_index][3])
        
        
        return self.otherCamIds_areas[area_index][3]
    
    def pop_person_inside_area(self, track_id, otherCamId_area):
        for idx, person_track in enumerate(self.persons_inside_area[otherCamId_area]):
            if person_track[0] == track_id:
                self.persons_inside_area[otherCamId_area].pop(idx)
                
    def clear_after_step(self):
        self.persons_inside_area.clear()
        

    
class Camera_area_forgetfulness(object):
    '''
    If BBox of somebody intesect with this area, it will disappear. 
    '''
    def __init__(self):
        self.otherCamIds_areas = []
        self.persons_inside_area = defaultdict(list)
    
    def add_new(self, area):   # (area_name, area)
        self.otherCamIds_areas.append(area)
        
    def bbox_belongs_to_zone(self, xy1xy2, eps=0):
        for otherCam_area in self.otherCamIds_areas:
            area_name = otherCam_area[0]
            zone = otherCam_area[1]
            if xy1xy2[0] > zone[0]-eps and xy1xy2[1] > zone[1]-eps and xy1xy2[2] < zone[2]+eps and xy1xy2[3] < zone[3]+eps:
                return True, otherCam_area[0]
        return False, -1
    
    def add_person(self, area_name, track_id):
        self.persons_inside_area[area_name].append(track_id)
    
    def get_all_persons_inside_area(self, area_name):
        return self.persons_inside_area[area_name]
    
    def get_amount_of_persons_inside(self, area_name):
        return len(self.persons_inside_area[area_name])


class Cassa_areas(object):
    def __init__(self):
        self.areas = []
        
    def add_new(self, area):
        self.areas.append(area)
    
    def get_cassa_area_by_cassa_idx(self, cassa_id):
        for area in self.areas:
            if area[1] == cassa_id:
                return area[0]
        return -1
    
class Exit_areas(object):
    def __init__(self):
        self.areas = []
        
    def add_new(self, area):
        self.areas.append(area)
    
    def get_cassa_area_by_exit_type(self, exit_type):
        for area in self.areas:
            if area[1] == exit_type:
                return area[0]
        return -1
    
    

class CameraInfo(object):
    '''
    Store info about different kind of areas, using defaultdict.
    Where defaultdict: Camera_id --> area_class
    
    '''
    
    def __init__(self):
        self.interseq_areas = defaultdict(Camera_interseq_areas)
        self.area_forgetfulness = defaultdict(Camera_area_forgetfulness)
        self.cassa_areas = defaultdict(Cassa_areas)
        self.exit_area = defaultdict(Exit_areas)
        
    def add_new_interseq_area(self, camera_id, other_cam_id, area, area_match_index, priority):
        self.interseq_areas[camera_id].add_new((other_cam_id, area, area_match_index, priority))
        
    def add_new_area_forgetfulness(self, camera_id, area_name, area):
        self.area_forgetfulness[camera_id].add_new((area_name, area))   
        
    def add_new_cassa_area(self, camera_id, cassa_id, area):
        self.cassa_areas[camera_id].add_new((area, cassa_id))
        
    def add_new_exit_area(self, camera_id, exit_type, area):
        self.exit_area[camera_id].add_new((area, exit_type))
        
            
    def get_interseq_areas(self, camera_id):
        return self.interseq_areas[camera_id].otherCamIds_areas
    
    def get_areas_forgetfulness(self, camera_id):
        return self.area_forgetfulness[camera_id].otherCamIds_areas
    
    def get_interseq_areas_with_id(self, camera_id, other_camera_id):
        if not self.interseq_areas[camera_id].otherCamIds_areas:
            return -1
        for other_cam_area in self.interseq_areas[camera_id].otherCamIds_areas:
            if other_cam_area[0] == other_camera_id:
                return other_cam_area[1]
        return -2
    
    def get_special_cassa_by_cam_idx(self, camera_id, cassa_id):
        return self.cassa_areas[camera_id].get_cassa_area_by_cassa_idx(cassa_id)
    
    
    
    def add_new_interseq_area_from_json(self, json):
        self.add_new_interseq_area(
            camera_id = json["camera_owner"],
            other_cam_id = json["camera_external"],
            area = json["area"],
            area_match_index = json["area_index"],
            priority = json["priority"]
        )
    def add_new_area_forgetfulness_from_json(self, json):
        self.add_new_area_forgetfulness(
            camera_id = json["camera_owner"],
            area_name = json["area_name"],
            area = json["area"]
        )
        
    def add_new_cassa_area_from_json(self, json):
        self.add_new_cassa_area(
            camera_id = json["camera_id"],
            cassa_id = json["cassa_id"], 
            area = json["area"]
        )
        
    def add_new_exit_area_from_json(self, json):
        self.add_new_exit_area(
            camera_id = json["camera_id"],
            exit_type = json["exit_type"], 
            area = json["area"]
        )
        