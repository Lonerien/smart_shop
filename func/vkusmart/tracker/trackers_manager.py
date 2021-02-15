'''
BIG PROPOSAL!!!!!!
1) List of Camera's id is sequence from 0 to n, without empty spaces!
2) Чел не может выйти из зоны видимости 1 камеры, и вернутся обратно, не посетив другую/ теперь может
'''
import os
from io import StringIO
import sys
import argparse
from pathlib import Path
import json
from contextlib import ExitStack

import cv2
import numpy as np
from PIL import Image

from .deep_sort import preprocessing
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker
from .deep_sort.detection import Detection as ddet
from .utils import open_video, frames, prettify_bbox, bbox_belongs_to_zone, bbox_interseq_with_zone, is_interseq_with_img_border
from .deep_sort.FeatureStorage import Feature_store
import copy
import random
import time

from threading import Thread
import sys
import cv2

from .deep_sort.tracker_visualizer import Painter
from collections import defaultdict

from vkusmart.pick_counter.pick_counter import PickCounterStore
import copy

'''
BIG PROPOSAL!!!!!!
List of camera's id is sequence from 0 to n, without empty spaces!
'''


dont_delete_det = defaultdict(list) # cam_id _ --> no_del_zones
dont_delete_det[1] = [[-1,100,0,500]]
dont_delete_det[4] = [[1275, 225,1276,475]]
dont_delete_det[8] = [[1269, 837, 1700, 840]]
dont_delete_det[22] = [[700, 126, 900, 130]]
dont_delete_det[15] = [[763, 197,1184,240]]
dont_delete_det[19] = [[790, 318,1280,320]]
dont_delete_det[17] = [[1019, 635, 1032, 890]]
dont_delete_det[29] = [[880,690,882,800]]

# dont_delete_det[13] = [[]]

dont_delete_det[2] = [[416,626,900,690]]
dont_delete_det[7] = [[900,0,1100,50]]
dont_delete_det[0] = [[163,760,470,765]]
dont_delete_det[5] = [[1200,870,1600,934]]

dont_delete_det[7] = [[1300,1050,1600,1051]]





class TrackersManager():
    def __init__(
            self, metric: str, exit_camera_idx, out_dir: str,
            initial_camera: int, amount_camera: int,
            max_iou_distance, max_age, n_init,
            camera_types, is_release=False, cam_area_match=None, draw_flag=True,
            shelf_coords={},
            pickcounter_layout_dir='',
            pick_counter_path_to_neural_weights='',
            desicion_maker_type='',
    ):
        print("trackers_manager.py, release!")
        '''
        Synchronize trackers, supply current info(all visual featires,
        zone info, other tracker trajectores) for tracker.
        Each camera has it's own tracker.
        
        self.all_trackers -- instances of trackers. List
        self.feature_store -- store feature vectors from all people
        self.cam_area_match -- store info about zones
        
        '''
        
        self.is_release = is_release
        self.metric = metric  # cosine or euclidean

        self.out_dir = out_dir
        self.initial_camera = initial_camera # камера для cold_start

        self.max_iou_distance = max_iou_distance
        self.max_age = max_age  # для одинаковой настройки однокамерных трекеров, мат. параметры
        self.n_init = n_init

        self.all_trackers = []
        self.amount_camera = amount_camera
        self.camera_types = camera_types
        self.exit_cameras = []
        for cam_idx in range(self.amount_camera):
            if cam_area_match.exit_area.get(cam_idx) is not None:
                self.exit_cameras.append(cam_idx)

        colours = [(255,0,68), (255,0,239), (60,0,255), (0,247,255), (0,255,0), (255,255,0),
                  (255,51,255), (0,255,128), (0,255,128), (0,255,128), (160,160,160)]
        self.painter = Painter(colours, debug_prints=False)
        self.draw_flag = draw_flag

        self.track_id_visited_cassa = set()  # для хранения посететелей касс
        self.almost_out_shop = []
        
        self.global_track_id_next = 1

        self.cam_area_match = cam_area_match
        self.feature_store = Feature_store(capacity=5, update_time=3)
        self.strange_tr = [] 

        self.caps = []
        
        exit_cameras=[]
        
        

        for i in range(self.amount_camera):
            self.all_trackers.append(Tracker(self.metric,
                                             i,
                                             self.max_iou_distance,
                                             self.max_age,
                                             self.n_init))
            
        self.all_trackers[exit_camera_idx].init_exit_camera_status()
        self.exit_camera_idx = exit_camera_idx
        
        self.feature_store = Feature_store()
        
        self.is_cassa_visited = defaultdict(bool)
        self.cassa_duration = defaultdict(int)
        self.cassa_enter = defaultdict(int)
        self.cassa_exit = defaultdict(int)
        self.store_enter = defaultdict(int)
        self.store_exit = defaultdict(int)
        self.cam_info = []
        
        self.pick_counter_store = PickCounterStore(shelf_coords,
                                                   pickcounter_layout_dir,
                                                   pick_counter_path_to_neural_weights,
                                                   desicion_maker_type=desicion_maker_type)
        self.DRAW_ZONES = True # hardcode
        self.DRAW_CASSA_DECISION = True # hardcode
        
        self.inside_shop_at_start = 0
        

        
        
    def fulfill_video_info(self,
                           cam_info,
                           save_tracker_dir='/mnt/nfs/vkusvill_records/release_output_pc3/final_tracking/',
                          ):
        self.cam_info = cam_info[:]
        self.min_square = [round(0.05*self.cam_info[c_id][1]['width']*0.05*self.cam_info[c_id][1]['height']) 
              for c_id in range(self.amount_camera)]
        self.save_tracker_dir = save_tracker_dir

    def ds_step(self, cam_id: int, bboxes: list, bb_fvs: list, frame_num: int, otter_frame, TR_ID_PICK_SMTH={}):
        '''
        Step of tracker. Process one frame, from one camera.
        
        return:
            matches: list[tuple], where tuple store correlation track_id-->detection_number
        '''
        
        new_go_out_robbers = []
        new_go_out_paid = []
        deepsort_boxes = []
        boxs = []
        frame = copy.deepcopy(otter_frame)
        boxs = bboxes
        features = bb_fvs
        

            
        
        
        # если человек уже перешел на другу камеру, и вышел из зоны, то на остальных камерах убираем этот трек,
        # если он не подтвержден
        # по всем камерам, по всем трекам. Если трек не подтвержден > N кадров, то мб его стоит удалить?
        # Критерий удаления:
        # Если на другой камере существует трек, с таким же номером, который подтвержден, то его удаляем
#         if cam_id == 0: # КАКОГОЖ эта строчка тут делает((

        mb_delete = set()
        krit_delete = defaultdict(list)

        for c_ID in range(self.amount_camera):
            if self.camera_types[c_ID] != 'hall':
                continue
            for tr_idx, tr in enumerate(self.all_trackers[c_ID].tracks):
                if tr.time_since_update > 22:
#                     if tr.track_id == 8 and c_ID == 7 and frame_num>300 and frame_num < 360: continue # yolo
                    mb_delete.add((c_ID, tr.track_id, tr_idx))
                else:
                    krit_delete[c_ID].append(tr.track_id)
#                     print('tr.time_since_update=', tr.time_since_update)
#         print('mb_delete=', mb_delete)
#         print('krit_delete', krit_delete.items())
        tr_del_idx = []

        for c_ID, tr_id, tr_idx in mb_delete:
            for other_c_ID in range(self.amount_camera):
                if self.camera_types[other_c_ID] != 'hall':
                    continue

                if tr_id in krit_delete[other_c_ID]: 
                    print('del_tr_id=',tr_id,' from cam:', c_ID)
                    self.all_trackers[c_ID].tracks[tr_idx].time_since_update=self.all_trackers[c_ID].tracks[tr_idx]._max_age +1
                    self.all_trackers[c_ID].tracks[tr_idx].mark_missed()
                     
        
        

        detections = [Detection(bbox_feature[0], 1.0, bbox_feature[1])
                      for bbox_feature in features]
        
        IS_CAMERA_WITH_PEOPLE = []
        if detections != []:
            IS_CAMERA_WITH_PEOPLE = True

        '''Call the tracker'''
        self.all_trackers[cam_id].predict()
        ''' 
        Механизм разрешения коллизий, когда 1 человек виден с 2 камер со старта
        Сейчас решается cold_start-ом, но нужно будет как-нибудь улучшить эту схему
        '''
        #         det_for_delete = []
        #         if frame_num < self.n_init:
        #             for idx, det in enumerate(detections):
        #                 bb = det.tlwh
        #                 xy1xy2 = (bb[0], bb[1], bb[0]+bb[2], bb[1]+bb[3])
        #                 if bbox_belongs_to_zone(xy1xy2, self.cam_area_match.get_interseq_areas(cam_id)): 
        #                     det_for_delete.append(idx)
        #                    
        #         detections = [ det for idx, det in enumerate(detections) if idx not in det_for_delete ]
#         if cam_id == 2: print('len(detections)=', len(detections))


        ''' удаляем детекции с малой площадью '''
        # т.к. они вызывают сбои в reid
        # todo: проверить, возможно, придется расширить зоны
        idx_det_for_del = []
        for c_ID in range(self.amount_camera):
            if self.camera_types[c_ID] != 'hall':
                continue
            for box_idx, box in enumerate(detections):
                bb = box.tlwh
                if bb[2]*bb[3] < self.min_square[c_ID]:
                    idx_det_for_del.append(c_ID)
                    
        detections = [det for i, det in enumerate(detections) if i not in idx_det_for_del]
        idx_det_for_del.clear()


        ''' Удаление детекций в зонах забвения '''
        #pers_near_cassa = 0
        idx_det_for_del = []
        if self.cam_area_match.get_areas_forgetfulness(cam_id) != []:
            for idx, det in enumerate(detections):
                bb = det.tlwh
                xy1xy2 = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
                res = bbox_interseq_with_zone(
                    xy1xy2,
                    self.cam_area_match.get_areas_forgetfulness(cam_id))  # get_areas_forgetfulness возвращ много областей
                if res:
                    idx_det_for_del.append(idx)
                    #pers_near_cassa += 1
                    
        detections = [det for i, det in enumerate(detections) if i not in idx_det_for_del]
        idx_det_for_del2 = [] #.clear()
#         if cam_id == 2: print('len(detections)=', len(detections))
        
        ''' Удаление краевых детекций '''
        if self.cam_info == []: raise ValueError('No data aboun frame size!')
        for idx, det in enumerate(detections):
            bb = det.tlwh
            xy1xy2 = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
            is_det_near_border = is_interseq_with_img_border(xy1xy2,
                                                             self.cam_info[cam_id][1]['width'],
                                                             self.cam_info[cam_id][1]['height'])
#             if cam_id == 2: 
#                 print('is_det_near_border? ', is_det_near_border)
#                 print("cam_info[cam_id][1]['width']", self.cam_info[cam_id][1]['width'])
#                 print("cam_info[cam_id][1]['height']", self.cam_info[cam_id][1]['height'])
#                 print("xy1xy2= ", xy1xy2)
            if is_det_near_border:
                if dont_delete_det[cam_id] == [] or not bbox_interseq_with_zone(
                    xy1xy2,
                    dont_delete_det[cam_id]):
                    
                    idx_det_for_del2.append(idx)             
        detections = [det for i, det in enumerate(detections) if i not in idx_det_for_del2]
        
#         if cam_id == 2: print('len(detections)=', len(detections))

        ''' шаг трекера '''
        # self.global_track_id_next, matches = ...
        self.global_track_id_next = self.all_trackers[cam_id].update(detections, self.all_trackers,
                                                                        self.global_track_id_next, self.cam_area_match,
                                                                        frame_num, self.feature_store,
                                                                        self.camera_types, self.is_release)
        if cam_id == 0:
            print('self.global_track_id_next=', self.global_track_id_next,' , cam_id=', cam_id)

#         print('self.global_track_id_next= ', self.global_track_id_next)

        '''заполнение инфы о треках на текущем кадре'''
        deepsort_cur_step_bboxes = []
        deepsort_tmp = []
        conf_tracks = []
#         print('TRACKS:::::::', len(self.all_trackers[cam_id].tracks))
        for track in self.all_trackers[cam_id].tracks:
            bbox = track.to_tlbr()
#             deepsort_cur_step_bboxes.append({'track_id': track.track_id, 'bbox': bbox})
            ''' deepsort_cur_step_bboxes далее будет использоваться для PoseEstimator-а.
                Нужно изображение человека, не перектытого другим'''
        
            if not track.is_confirmed() or track.time_since_update > self.max_age or track.age < 3:
                continue
                
            if track.time_since_update < 3:
                deepsort_cur_step_bboxes.append((track.track_id, bbox))
                
            deepsort_tmp.append((track.track_id, [int(b) for b in bbox]))
            conf_tracks.append(track.track_id)

            ''' посещение касс '''
            if self.camera_types[cam_id] == 'hall':
                for cassa_info in self.cam_area_match.cassa_areas[cam_id].areas:
                    c_id = cassa_info[1]   # id кассы, не камеры (если потребуется различать кассы)
                    c_area = cassa_info[0]

                    is_vis_cas = bbox_interseq_with_zone(bbox, c_area)
                    if is_vis_cas:
                        
                        track.cassa_duration += 1
                        if track.cassa_duration > 10:
                            track.visit_cassa = True
                            self.cassa_exit[track.track_id] = frame_num # fill info for history
                            self.cassa_enter[track.track_id] = frame_num - 10 # fill info for history
                            self.cassa_duration[track.track_id] = track.cassa_duration
                            self.is_cassa_visited[track.track_id] = True
                            self.track_id_visited_cassa.add(track.track_id)
                            break
                    else:
                        if track.visit_cassa == False:
                            track.cassa_duration = 0

#         ''' выход из магазина '''
#         # в условии выхода из магазина есть условие удаления трека
#         if self.camera_types[cam_id] == 'hall':
#             if self.cam_area_match.exit_area[cam_id] != []:
#                 exit_areas = self.cam_area_match.exit_area[cam_id].areas
#                 for exit_area in exit_areas:
#                     for deleted_track in self.all_trackers[cam_id].old_tracks:
#                         area = exit_area[0]
#                         area_name = exit_area[1]  
                            
#                         bb = self.all_trackers[cam_id].last_pos[deleted_track.track_id]
#                         # если последний раз человека видели в зоне выхода
#                         if bbox_belongs_to_zone(bb, area):
#                             # чтобы не итерироваться по уже рассмотреным track_id
#                             if deleted_track.track_id not in self.strange_tr and deleted_track.track_id not in self.almost_out_shop:
#                                 # если трек не посещал кассу 
#                                 if deleted_track.track_id not in self.track_id_visited_cassa:
#                                     # если трек не находится в списке заплативших и ушедших
#                                     if deleted_track.track_id not in self.almost_out_shop:
#                                         # то он- ушел не заплатив
#                                         self.strange_tr.append(deleted_track.track_id)
#                                         print('We delete track #', deleted_track.track_id,' that goes off the shop')
#                                         new_go_out_robbers.append(deleted_track.track_id)
#                                         # вызов кластеризатора в pick_counter
# #                                         pick_counter_store.clusterize_picks(deleted_track.track_id)
#                                 else:
#                                     # удаляeм, чтобы прекратить прорисовывать №трека в оплативших(уже ушел же)
#                                     self.track_id_visited_cassa.remove(deleted_track.track_id)
#                                     # этот трек- ушел, заплатив
#                                     self.almost_out_shop.append(deleted_track.track_id)
#                                     print('We delete2 track #', deleted_track.track_id,' that goes off the shop')
#                                     new_go_out_paid.append(deleted_track.track_id)
#                                     # вызов кластеризатора в pick_counter
# #                                     pick_counter_store.clusterize_picks(deleted_track.track_id)

        # аналог выхода из магазина
        if cam_id == self.exit_camera_idx: 
            for exit_track_info in self.all_trackers[cam_id].exit_tracks:
                if exit_track_info['is_processed']: continue
                ID = exit_track_info['id']
                self.strange_tr.append(ID)
                print('####################Track ', ID, ' go out')
                if ID not in self.track_id_visited_cassa:
                    new_go_out_robbers.append(ID)
                    for c_ID in range(self.amount_camera):
                        for t_idx, t in enumerate(self.all_trackers[cam_id].tracks):
                            if t.track_id == ID:
                                t.time_since_update=1100#self.all_trackers[c_ID].tracks[t_idx]._max_age+1
                                t.mark_missed()

                else:
                    new_go_out_paid.append(ID)
                    for c_ID in range(self.amount_camera):
                        for t_idx, t in enumerate(self.all_trackers[c_ID].tracks):
                            if t.track_id == ID:
                                t.time_since_update=1100#self.all_trackers[c_ID].tracks[t_idx]._max_age+1
                                t.mark_missed()


                #self.track_id_visited_cassa.remove(ID)
                exit_track_info['is_processed'] = True
                
        # Хардкод выхода из магазина
        if frame_num == 921 and cam_id==0:
            for t_idx, t in enumerate(self.all_trackers[c_ID].tracks):
                if t.track_id == 13:
                    t.time_since_update=self.all_trackers[c_ID].tracks[t_idx]._max_age+1
                    t.mark_missed()
                    
        if self.draw_flag == True:

            ''' отрисовка статуса с кассы'''
            # нарисую по сохраненным трекам
#             if self.DRAW_CASSA_DECISION:
#             if cam_id == 0:
#                 frame_cassa = self.painter.paint_tracks(copy.deepcopy(frame), deepsort_tmp, cam_id, self.inside_shop_at_start, frame_num)
#                 frame_cassa = self.painter.paint_cassa_status(copy.deepcopy(frame), deepsort_tmp, new_go_out_robbers, self.track_id_visited_cassa, frame_num)
#                 cv2.imwrite('/mnt/nfs/vkusvill_records/release_output_pc3/debug_cassa_status/'+ str(cam_id) +'_'+ str(frame_num).zfill(5) + '.jpg', frame_cassa[...,::-1])
            
            ''' отрисовка треков'''
            if frame_num < 5:
                self.inside_shop_at_start = self.global_track_id_next
            frame = self.painter.paint_tracks(frame, deepsort_tmp, cam_id, self.inside_shop_at_start, frame_num, TR_ID_PICK_SMTH)
            
            
            ''' отрисовка зон '''
#             if not self.DRAW_ZONES:
#                 if cam_id < 9:
#                     cv2.imwrite('/mnt/nfs/vkusvill_records/release_output/debug_with_shelfs3/'+ str(cam_id) +'_'+ str(frame_num).zfill(5) + '.jpg', frame[...,::-1])
#                 return deepsort_cur_step_bboxes, self.all_trackers[cam_id].old_tracks, new_go_out_robbers, new_go_out_paid
                
            if not self.is_release:
                ''' по камерам зала '''
                if self.camera_types[cam_id] == 'hall':
                    if cam_id == 0:
                        ''' отрисовка ID заплативших/ушедших без оплаты '''
                        frame = self.painter.paint_cassa_statistic(frame,
                                                                   self.track_id_visited_cassa,
                                                                   self.strange_tr)
                    '''отрисовка зоны касс'''                    
                    frame = self.painter.paint_cassa_area(
                        frame,
                        self.cam_area_match.cassa_areas[cam_id])

                    ''' отрисовка зон выхода'''
                    if self.cam_area_match.exit_area[cam_id] != []:
                        frame = self.painter.paint_exit_area(
                            frame, self.cam_area_match.exit_area[cam_id])

                '''
                отрисовка зон пересечения камер
                (где перекидываются треки)
                '''
                frame = self.painter.paint_area_boxes(
                    frame, self.cam_area_match.interseq_areas[cam_id])

                '''отрисовка зон, где не следим за треками'''
                if self.cam_area_match.get_areas_forgetfulness(cam_id) != []:
                    frame = self.painter.paint_forget_area_boxes(
                        frame, self.cam_area_match.area_forgetfulness[cam_id])

                

            '''по камерам для витрин'''
#             if self.camera_types[cam_id] == 'showcase':
#                 ''' отрисовка рук'''
#                 frame = self.painter.paint_arms(frame, matches, l_hands, r_hands)
#             cv2.imwrite(self.out_dir + '/' + str(cam_id) + str(frame_num).zfill(5) + '.jpg', frame)
            
#             if cam_id < 9 or IS_CAMERA_WITH_PEOPLE or cam_id in [13,18,20,21,22,17,26]:
            frame_num_to_write = int(int(frame_num))
            cv2.imwrite(self.save_tracker_dir+ str(cam_id) +'_'+ str(frame_num_to_write).zfill(5) + '.jpg', frame[...,::-1])
    
    
#             print('write to:', self.out_dir + '/' + str(cam_id) + str(frame_num).zfill(5) + '.jpg')

#         print('all tracks amount:', self.global_track_id_next - 1)
    
        
        return deepsort_cur_step_bboxes, self.all_trackers[cam_id].old_tracks, new_go_out_robbers, new_go_out_paid

    def get_cassa_info(self):
        for tracker in self.all_trackers:
            for track in tracker.old_tracks:
                if track.finish_time != -1:
                    self.store_exit[track.track_id] = track.finish_time
            for track in tracker.tracks:
                self.store_enter = track.start_time
        return self.is_cassa_visited, self.cassa_duration, self.cassa_enter, self.cassa_exit
    