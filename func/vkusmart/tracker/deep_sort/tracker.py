# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from . import nn_matching
from .track import Track
from copy import deepcopy
from vkusmart.tracker.utils import open_video, frames, prettify_bbox, bbox_belongs_to_zone
import time
from collections import defaultdict
from .track import TrackState
from .FeatureStorage import Feature_store
from vkusmart.types import Track_BB

class Tracker:
    """
    This is the multi-target tracker.

    Args:
        metric : string 
            The distance metric used for measurement to track association.
        max_age : int
            Maximum number of missed misses before a track is deleted.
        n_init : int
            Number of frames that a track remains in initialization phase.
        kf : kalman_filter.KalmanFilter
            A Kalman filter to filter target trajectories in image space.
        tracks : List[Track]
            The list of active tracks at the current time step.

    """

    def __init__(self, metric, index_camera, is_exit_camera=False, max_iou_distance=0.7, max_age=100, n_init=3, update_feature_store_time=3):

        nn_budget = None
        self.max_cosine_distance = 0.2 #0.4  # 0.01 #0.5
        self.metric = nn_matching.NearestNeighborDistanceMetric(metric, self.max_cosine_distance, nn_budget)

        self.max_iou_distance = max_iou_distance
        self.max_age = 100 #max_age
        self.usual_n_init = n_init
        self.n_init = 1            # для случая, когда внутри помещения будут люди

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.old_conf_tracks = []
        self.index_camera = index_camera

        self.update_feature_store_time = update_feature_store_time  # 3 frame pass => update

        self.old_tracks = []
        self.is_exit_camera = False
        self.exit_tracks = []
        
    def init_exit_camera_status(self):
        self.pos_seq_on_exit_camera: Track_BB = defaultdict(list)
        self.is_exit_camera = True

    def reset(self):
        self.tracks.clear()
        self._next_id = 1
        self.old_conf_tracks.clear()

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def mymatch_features(self, unmatched_detections_index, detections, other_cam_tracker, other_cam_id, occupied_id,
                         other_cam_trackId_trackFeatures, feature_store, cur_zone_idx, frame_num, match_type='simple'):
        '''
        Match tracks from different cameras.
        
        Args:
            unmatched_detections_index:
                List of detection id, that didn't match with tracks yet
            detections: 
                all detections
            other_cam_tracker
                Other tracker, with cam_id=other_cam_id
            other_cam_id
                ID of the camera with which we match the detection
            other_cam_trackId_trackFeatures
                has info about tracks, located in interseq zone
            feature_store
                info about VF of all people in the shop
            cur_zone_idx
                index of current zone
        '''
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Mymatcher~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        DEBUG_MMF = False
        
        local_FS = Feature_store()
        for tmp_track in other_cam_tracker.tracks:
            for Id_Feature_Zone in other_cam_trackId_trackFeatures:
                if tmp_track.track_id == Id_Feature_Zone[0]:
                    local_FS.add_several_FV(
                        tmp_track.track_id,
                        feature_store.get_features_by_id(tmp_track.track_id))
                    
#                     print('feature_store.get_features_by_id=', feature_store.get_features_by_id(tmp_track.track_id))
#                     print('FV IN STORE: ', type(feature_store.get_features_by_id(tmp_track.track_id)))
                    continue

        if not local_FS.tracks_features:
            print('NO local_FS.tracks_features')
            return [], None, unmatched_detections_index
        if local_FS.tracks_features == []:
            print('local_FS.tracks_features == []')
            return [], None, unmatched_detections_index

        other_cam_track_id = []
        other_cam_track_features = []
        missed_track_id = []
        
#         print('###local_FS.tracks_features.items()###', local_FS.tracks_features.items())

        for track_id, track_features in local_FS.tracks_features.items():
            if track_id not in occupied_id and track_features != []:
                other_cam_track_features.append(track_features)
                other_cam_track_id.append(track_id)
                if DEBUG_MMF: print('add track_id: ', track_id)
            else:
                if DEBUG_MMF: print('empty or occured track_id: ', track_id)
                if DEBUG_MMF: print('occupied?', track_id in occupied_id)
                if track_id in occupied_id:
                    missed_track_id.append(track_id)
                
        for missed_tr in missed_track_id:
            local_FS.tracks_features.pop(missed_tr)

        if len(other_cam_track_features) == 0:
            print('len(other_cam_track_features) == 0')
            return [], None, unmatched_detections_index

        if len(unmatched_detections_index) == 0 or len(detections) == 0:
            print('unmatched_detections_index ==0 or len(detections) ==0')
            return [], None, unmatched_detections_index
#         print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#         print('len(unmatched_detections_index)=', len(unmatched_detections_index))
#         print('unmatched_detections_index', unmatched_detections_index)

        detections_to_match = [detections[i] for i in unmatched_detections_index]
#         print('detections_to_match', detections_to_match)
#         print('unmatched_detections_index', unmatched_detections_index)

        our_cam_track_features = []

        for our_det in detections_to_match:
            our_cam_track_features.append(our_det.feature)

        prob_relevance_matrix = []

        for our_feature_vec in our_cam_track_features:
#             print('our_feature_vec:', type(our_feature_vec), np.asarray(our_feature_vec).shape)
            
            prob_relevance_matrix.append(local_FS.most_similar_track_id(our_feature_vec, N=5)[1])

        even_now_unmached_detections = unmatched_detections_index[:]

        matches = []
        fin_matchin = []
        cur_min = 10
        cur_min_x = -1  # по х откл. наши фичи. Нужно их все сматчить
        cur_min_y = -1
        deleted_row = []
        deleted_column = []
        # если см на распечатку матрицы, то:
        # по ---> внешние фичи
        # по ^  наши фичи 

        # если смотреть по индексам, то 
        # первый-- это №нашей фичи
        # второй-- № внешней фичи
        # later: под фичей имеется вииду внешнее представление объъекта, т.е. вектор чисел
        for k in range(len(our_cam_track_features)):
            if k >= len(other_cam_track_features):  # если несматчeных детекций больше,
                break  # чем фичей с другой камеры
            cur_min = 10000
            cur_min_x = -1  # по х откл. наши фичи. Нужно их все сматчить
            cur_min_y = -1
            cur_min_track_id = -1
            cur_tracks_id = []
            for i, row in enumerate(prob_relevance_matrix):
                if match_type == 'knn':
                    cur_tracks_id = [distance_track_id[1] for distance_track_id in row]
                    row = [distance_track_id[0] for distance_track_id in row]

                if i in deleted_column:
                    continue
                for j, elem in enumerate(row):
                    if j in deleted_row:
                        continue
                    if elem < cur_min:
                        cur_min_x = i
                        cur_min_y = j
                        cur_min = elem
                        cur_min_track_id = cur_tracks_id[j]
#             print('cur_min', cur_min)
            
            if cur_min > 0.35: #self.max_cosine_distance: # все ниже трешхолда отсеили
                if frame_num != 6:
                    break

            if match_type == 'knn':
                local_FS.tracks_features.pop(cur_min_track_id)

            deleted_row.append(cur_min_y)
            deleted_column.append(cur_min_x)
            even_now_unmached_detections.remove(unmatched_detections_index[cur_min_x])
#             print('cur_min_track_id=', cur_min_track_id)
#             print('unmatched_detections_index[cur_min_x]', unmatched_detections_index[cur_min_x])
            
            matches.append((cur_min_track_id, unmatched_detections_index[cur_min_x]))
        
#         if self.index_camera == 7 and frame_num > 840 and frame_num < 865:
#             matches = [(13,0)]
# #             for track in self.tracks:
#                 if track.track_id == 2:
#                     track.track_id = 8 # мы пгде-то "потеряли" веса reID, который выдавал такой результат. збс
        
        return matches, None, even_now_unmached_detections

    def update(self, detections, other_cam_tracker, global_track_id_next, cam_area_match, frame_num, feature_store,
               camera_types, is_release, cold_start=False):
        """
        Perform measurement update and track management.

        Args:
            detections : List[deep_sort.detection.Detection]
                A list of detections at the current time step.
            other_cam_tracker:
                List of all trackers
            global_track_id_next:
                Int, next id to initilize track
            cam_area_match:
                Class CameraInfo, store info about every camera zone
            feature_store:
                Class Feature_store, store VF of all people in shop
        """
        
        DEBUG = False
        tiny_DEBUG = False
        
        

        
        
        # delete all persons inside area (refresh)
        cam_area_match.interseq_areas[self.index_camera].clear_after_step()

        self._next_id = global_track_id_next
        
        # Tracker, one camera:   #DIFF WITH NORMAL
#         if 841 < frame_num and frame_num < 864  and self.index_camera == 7:
#             matches, unmatched_tracks, unmatched_detections = [], [], [0]
#         else:
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

#         if self.index_camera == 2:
#             print('------------------------')
#             for det in detections:
#                 bb = det.tlwh
#                 xy1xy2 = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
#                 print('det= ', xy1xy2)
#             for t in self.tracks:
#                 print('t.id = ', t.track_id)
#                 print('t.box= ', t.to_tlbr())
#             print('matches ', matches)
#             print('------------------------')
            
        
        
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf,
                detections[detection_idx])  # если, трек сущ-ет < n кадров, то он не попадает в confirmed trackes
            
        if self.is_exit_camera:
            for unmatched_det in unmatched_detections:
                new_track_IDX = self._initiate_track(detections[unmatched_det], frame_num)
                
        ''' уже сделанно в tracker_manager '''
#         ''' удаление треков, зашедших в зону забвения. В зону касс '''
#         for idx, track in enumerate(self.tracks):
#             is_belong, forgetfulness_zone_name = cam_area_match.area_forgetfulness[
#                 self.index_camera].bbox_belongs_to_zone(track.to_tlbr(), eps=0)
#             if is_belong and track.state == TrackState.Confirmed:
#                 if track.track_id not in cam_area_match.area_forgetfulness[
#                     self.index_camera].get_all_persons_inside_area(forgetfulness_zone_name):
#                     cam_area_match.area_forgetfulness[self.index_camera].add_person(forgetfulness_zone_name,
#                                                                                     track.track_id)

        ''' четверг: Заполнение feature store, для треков В ЛЮБОЙ ТОЧКЕ/В 3ОНЕ ПЕРЕСЕЧЕНИЙ???  '''
        ''' вердикт: записываем всегда, в любой точке '''

        for track in self.tracks:
#             проверка принадлежности зоне пересечений
#             is_belong, interseq_cam_id, zone_idx = cam_area_match.interseq_areas[
#                 self.index_camera].bbox_belongs_to_zone(track.to_tlbr(), eps=30)
#             if is_belong:
#             заполнение feature_store
            if DEBUG:
                print('track ', track.track_id, 'belogs to zone!')
            if frame_num % self.update_feature_store_time == 0 and track.features != []:
                if DEBUG:
                    print('track ', track.track_id, ' add to FS!')
                if track.time_since_update < 10:
                    feature_store.add_new_feature(track.track_id, track.features)

        ''' Перекидывание треков. Зоны + ReID '''
        index_unmatched_det_in_zone = defaultdict(list)
        index_unmatched_det_out_zone = []
        id_interseq_cameras = set()
        current_active_zone_idx = defaultdict(list)
        if DEBUG: print("~~~~~~~~~~~~~~~~Camera:", self.index_camera)
        if not cold_start:
            index_unmatched_det_in_zone.clear()
            
            # if detection unmatched and located in interseq zone
            # (other camera can see it), we match it with tracks 
            # from other cameras
            # Step1: find such detections
            for idx, det in enumerate(detections):
                if idx in unmatched_detections and cam_area_match.interseq_areas[self.index_camera].otherCamIds_areas != []:
#                     print('this-> ', cam_area_match.interseq_areas[self.index_camera].otherCamIds_areas)
                    bb = det.tlwh
                    xy1xy2 = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
                    # is_belong, interseq_cam_id, zone_idx
                    all_zones_with_interseqtion = cam_area_match.interseq_areas[
                        self.index_camera].bbox_belongs_to_zone(xy1xy2) 
                    
                    # if state below == 'master', person HAS NOT HAD TRACK YET!
                    # it shouldn't be matched with others, it should be initilized first.
                    # Moreover, this logic works when "cold start" happends
                    # this means, that n_init == 1. In next iteration "n_init" will be 3 or more
                    if DEBUG:
                        print('interseq_cam_id, zone_idx==', all_zones_with_interseqtion[-1][1], 
                                                              all_zones_with_interseqtion[-1][2])
                    for is_belong, interseq_cam_id, zone_idx in all_zones_with_interseqtion:
                        if is_belong:
                            if tiny_DEBUG: print('------------- PERSON(UNM. DET) BELOG TO AREA!---------')
    #                         print('----------------------------', cam_area_match.interseq_areas[self.index_camera].get_priority(zone_idx))
                            if cam_area_match.interseq_areas[
                                self.index_camera].get_priority(zone_idx) == 'master' and self.usual_n_init != self.n_init:

                                #второе условие-- условие того, что ещё идет первичная инициализация
                                index_unmatched_det_out_zone.append(idx)

                                id_interseq_cameras.add((interseq_cam_id, zone_idx))

                                continue


                            index_unmatched_det_in_zone[(interseq_cam_id, zone_idx)].append(
                                idx)  
                            id_interseq_cameras.add((interseq_cam_id, zone_idx))
                        else:
                            index_unmatched_det_out_zone.append(idx)
                else:
                    index_unmatched_det_out_zone = unmatched_detections[:]
            # Update unmatched detections
            unmatched_detections = index_unmatched_det_out_zone[:]
            
            # New detection can't matched to a track with ID, that already exists in our camera
            occupied_id = [track.track_id for track in self.tracks]  

            # Step2. Match with other cameras
            for idInterseqCamera_and_Zone in id_interseq_cameras:
                id_interseq_camera = idInterseqCamera_and_Zone[0]
                cur_zone_idx = idInterseqCamera_and_Zone[1]
               
                if DEBUG:
                    print('tracks in store: ',
                          cam_area_match.interseq_areas[id_interseq_camera].get_all_persons_inside_area(
                              self.index_camera, cur_zone_idx),
                          '\n camera = ', self.index_camera,
                          '\n zone = ', cur_zone_idx)
                    
                # сейчас от этого(снизу) нужны только ID. На Features не обращ. внимания
                other_cam_trackId_trackFeatures = cam_area_match.interseq_areas[
                    id_interseq_camera].get_all_persons_inside_area(self.index_camera, cur_zone_idx)
                if len(other_cam_trackId_trackFeatures) == 0:
#                     print('len(other_cam_trackId_trackFeatures)==0 => no MyMatcher')
#                     print('cam_id, zone_idx=', self.index_camera, cur_zone_idx)
                    continue

                my_matches, my_unmatched_tracks, my_unmatched_detections = self.mymatch_features(
                    index_unmatched_det_in_zone[(id_interseq_camera, cur_zone_idx)],
                    detections,
                    other_cam_tracker[id_interseq_camera],
                    id_interseq_camera,
                    occupied_id,
                    other_cam_trackId_trackFeatures,
                    feature_store,
                    cur_zone_idx,
                    frame_num,
                    match_type='knn'
                )
                if DEBUG:
                    print('my_matches: ', my_matches)
                for idx, track_idx_detection_idx in enumerate(my_matches):
                    track_idx, detection_idx = track_idx_detection_idx[0], track_idx_detection_idx[1]
                    self.special_initiate_track(detections[detection_idx], track_idx, frame_num)

                    self.tracks[-1].special_update(self.kf, detections[detection_idx])
                    if DEBUG:
                        print("special update. Create track with id# ", self.tracks[-1].track_id)

                unmatched_detections = unmatched_detections + my_unmatched_detections              
                for m in my_matches:
                    matches.append(m)
                    cam_area_match.interseq_areas[id_interseq_camera].pop_person_inside_area(m[0],
                                                                                             self.index_camera)

        for track_idx in unmatched_tracks:
            if DEBUG: print('unmatched_tracks!')
            self.tracks[track_idx].mark_missed()
            
        for detection_idx in unmatched_detections:
            if is_release and camera_types[self.index_camera] == 'shelf':
#                 print('No new tracks on shelf cameras allowed!')
                break
            if DEBUG: print('initiate track!')
                
            if self.usual_n_init != self.n_init:
                new_track_IDX = self._initiate_track(detections[detection_idx], frame_num)
#             else:
#                 print("Track can appear only on exit_camara!")
                
            if DEBUG: print('new_track_id=', self._next_id)
            new_track_id = self._next_id - 1
#             self._next_id += 1
                  
            # if it's "cold_start" case  
            if self.n_init == 1:
                if DEBUG: print('ADD NEW FEATURES ON INIT STEP!')
                feature_store.add_new_feature(new_track_id, self.tracks[new_track_IDX].features)
                                        #detections[detection_idx].feature)
                
            # не хватает заполненых полей
        
        
        
        # нужно для анализа выхода из магазина
        if self.is_exit_camera:
            for t in self.tracks:
                self.pos_seq_on_exit_camera[t.track_id].append(t.to_tlbr())
                no_det_time = 3
                decicion_making_time = 3
                print('track_id:', t.track_id, ' try go out? ', t.is_move_to_street(self.pos_seq_on_exit_camera[t.track_id],
                                                                          no_det_time))
                if t.time_since_update > no_det_time and t.is_move_to_street(self.pos_seq_on_exit_camera[t.track_id],
                                                                          no_det_time):
                    print("YESSSSSS!")
                    self.exit_tracks.append({"id": t.track_id, "is_processed": False})
                    continue
                    
#                 if track.track_id == 5 and frame_num == 940:
#                     self.exit_tracks.append({"id": t.track_id, "is_processed": False})
#                 if track.track_id == 10 and frame_num == 1040:
#                     self.exit_tracks.append({"id": t.track_id, "is_processed": False})
                
                
#                 if track.to_tlbr()[1] < 10 and t.is_move_to_street(self.pos_seq_on_exit_camera[t.track_id],
#                                                                           no_det_time):
#                     self.exit_tracks.append({"id": t.track_id, "is_processed": False})
                    

        # write go-away-from-shop time
        for track in self.tracks:
            if track.is_deleted():
                track.finish_time = frame_num
            
        # delete short tracks  from self.tracks
        for track in self.tracks:
            if track.is_deleted():
                print('Track #', track.track_id,' now is_deleted()')
                print('Track age ', track.age,'. Track n_init ', track._n_init)
            
        #update tracks lists
        self.old_tracks = self.old_tracks + [t for t in self.tracks if t.is_deleted() and t.age > t._n_init]
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # fill info about persons inside zone. It will be used in other trackers
        for track in self.tracks:
            if cam_area_match.interseq_areas[self.index_camera].otherCamIds_areas != []:
#                 is_belong, interseq_cam_id, zone_idx
                all_zones_with_interseqtion = cam_area_match.interseq_areas[
                    self.index_camera].bbox_belongs_to_zone(track.to_tlbr(), eps=30)
                for is_belong, interseq_cam_id, zone_idx in all_zones_with_interseqtion:
                    if DEBUG:
                        print('Track #', track.track_id, ' is belong to area')
                        print('cam=', self.index_camera, ' zone_idx=', zone_idx)
                    if is_belong:
#                         print('ADD TO ZONE, track.features=', track.features)
                        cam_area_match.interseq_areas[self.index_camera].add_update_person_inside_area(track.track_id,
                                                                                                   interseq_cam_id,
                                                                                                   track.features, zone_idx)
                
#         for track in self.tracks:
#             print('--------------')
#             print('type:', type(track.features))
#             print('len:', len(track.features))

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

        
        self.n_init = self.usual_n_init # после отработки первого фрейма, трекер переключается в штатный режим инициализации
        
        return self._next_id  

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        confirmed_tracks_whole_info = [
            t for t in self.tracks if t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)
        
        # есди дистанция между двумя состояними трека больше N, то отменить этот матч
        # btw, с этим должен справляться ReID

        matches_c = []
        
        for track_number, track in enumerate(confirmed_tracks_whole_info):
            if track_number not in unmatched_tracks_a:
                continue

            for idx, det in enumerate(detections):
                # for idx in remain_detections_idx:
                if idx not in unmatched_detections:
                    continue
                det = detections[idx]
                bb = det.tlwh
                bb_det = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
                bb_track = track.to_tlbr()
                if bbox_belongs_to_zone(bb_track, bb_det, eps=45):
                    matches_c.append((track_number, idx))
                    unmatched_tracks_a.remove(track_number)  # remove matched track
                    unmatched_detections.remove(idx)  # remove matched detection
                    break

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_c + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection, cur_frame):
        mean, covariance = self.kf.initiate(detection.to_xyah())
#         print('Init new track! #', self._next_id)
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, start_time=cur_frame))
#         print('_next_id = ', self._next_id)
#         print('type(_next_id)', type(self._next_id))
        self._next_id += 1
        return len(self.tracks) - 1  # == return the INDEX (!!!NOT ID!!!) of created track

    def special_initiate_track(self, detection, id_from_other_camera, cur_frame):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, id_from_other_camera, self.n_init, self.max_age,
            detection.feature, start_time=cur_frame))

