from typing import Union, List, Tuple, Any, Dict, DefaultDict
from datetime import datetime
from pathlib import Path


from collections import defaultdict
from scipy.spatial import distance
from vkusmart.utils import one_dim_clasterize
import numpy as np
import csv
import os
from numpy import linalg as LA
import jsonpickle
import json
from collections import Counter

from vkusmart.pick_counter.evristic_method import EvristicMethod
from vkusmart.pick_counter.neural_method_v1 import PickNetClassifier as PickNetV1

DEBUG = False
IS_HARDCODE = False #True

# class HARDCODE_INFO:
#     shift=30
#     cameras_shift = [404+shift, 384+shift, 312+shift, 386, 400+shift,183,385,381,671,473+shift,386,389,166,405,187,411,253,480,256,481,699,236,12,236,671,263,676,458,461,352,145,174]
#     pc_res = {
#         '3':{
#             '22':{
#                 'left':[[1900,0,0]],
#                 'right':[]
#             }
#         },
#         '8':{
#             '13':{
#                 'left':[],
#                 'right':[
#                 [1396,1,0],
#                 [1580,0,1]]
#             },
#             '15':{
#                 'left':[],
#                 'right':[[2980,1,0],
#                 [3128,1,0],
#                 [3379,1,0]]
#             },
#             '17':{
#                 'left':[],
#                 'right':[[3053,1,0],
#                 [3199,1,0],
#                 [3451,1,0]]
#             },
#             '18':{
#                 'left':[],
#                 'right':[[1848,1,0]]
#             },
#             '20':{
#                 'left':[],
#                 'right':[
#                 [1687,1,0],
#                 [1875,0,1]]
#             },
#             '21':{
#                 'left':[],
#                 'right':[[1849,1,0]]
#             },
#             '22':{
#                 'left':[],
#                 'right':[[2117,1,0]]
#             },
#         }
#     }


def calc_dist(point1, point2):
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    return (x ** 2 + y ** 2) ** 0.5


class Velocity:
    def __init__(self):
        self.x_component: float = 0.0
        self.y_component: float = 0.0
        self.module: float = 0.0
        self.module_to_shelf: float = 0.0

    def dist_point_line(self, p, line):
        p1 = np.asarray([line[0], line[1]])
        p2 = np.asarray([line[2], line[3]])
        p3 = np.asarray(p)
        return LA.norm(np.cross(p2 - p1, p1 - p3)) / LA.norm(p2 - p1)

    def calc_vel(self, prev_arm_center, cur_arm_center, time_duration, shelf_line=[0, 0, 0, 1]):
        self.x_component = prev_arm_center[0] - cur_arm_center[0]
        self.y_component = prev_arm_center[1] - cur_arm_center[1]
        dist = ((self.x_component ** 2) + (self.y_component ** 2)) ** 0.5
        self.module = dist / (time_duration * 1.)
        if DEBUG: print('self.x_component=', self.x_component)

        dist_to_line_prev = self.dist_point_line(prev_arm_center, shelf_line)
        dist_to_line_cur = self.dist_point_line(cur_arm_center, shelf_line)

        self.module_to_shelf = dist_to_line_prev - dist_to_line_cur
        
        
class Distance: # AI
    def __init__(self):
        self.x_component: float = 0.0
        self.y_component: float = 0.0        
        self.distance_to_shelf: float = 0.0
        self.distance_to_hip: float = 0.0
            
    def dist_point_line(self, p, line): # TODO: отнаследовать метод (автор не силён в ООП)
        p1 = np.asarray([line[0], line[1]])
        p2 = np.asarray([line[2], line[3]])
        p3 = np.asarray(p)
        return LA.norm(np.cross(p2 - p1, p1 - p3)) / LA.norm(p2 - p1)
    
    def calc_distns(self, cur_arm_center, cur_shelf_coords=[0, 0, 0, 1], hip_point=[0.5, 0.5]):
        
        self.distance_to_shelf = self.dist_point_line(cur_arm_center, cur_shelf_coords)
        
        self.x_component = hip_point[0] - cur_arm_center[0]
        self.y_component = hip_point[1] - cur_arm_center[1]
        self.distance_to_hip = ((self.x_component ** 2) + (self.y_component ** 2)) ** 0.5
        if DEBUG: print('self.x_component=', self.x_component)

        
        

class PickCounter:
    def __init__(self,
                 track_id: int,
                 timestamp: int,
                 time_step: int,
                 goods_already_picked: int,
                 desicion_maker_type: str = 'evristic',
                 streak: int = 12,
                 good_in_streak: int = 10,
                 shelf_coords: list = [0, 0, 0, 0],
                 pickcounter_layout_dir='/mnt/nfs/vkusmart/master/data/session_test/test_pick_counter',
                 path_to_neural_weights='',
                 camera_idx=-1
                 ):
        '''
        PickCounter -- класс, ответственный за подсчет и хранение взятых продуктов.
        Используется в PickCounterStore (см внизу)
        Отвечает за одного человека.
        Порождается при появлении человека на камере, промаркированной как 'shelf_camera'
        Уничтожается при уходе человека с камеры


        shelf_coords -- координаты зоны полки
        desicion_maker_type -- эвристический или RNN

        стандартное поведение:
        1-ый кадр:
            PickCounterStore --> создать новый PickCounter для трека ID на камере camera_idx
                   __init__ (инициализация полей) -->
                   _init_evristic_method
            PickCounterStore --> вызов update() у PickCounter

                   update() --> видит что только появился. Заполняет инфу о руках на текущем кадре,
                   завершается без подсчета скорости. Возвращает кол-во товаров у человека,
                   когда он только пришел на эту камеру

        N-ый кадр:
            PickCounterStore --> вызов update() у PickCounter
                   update()  (обновление координат и ArmState, и доп инфы вроде скорости) -->
                   calc_pick_count() считает кол-во взятых товаров на текущем фрейме
                   с пом-ю методов ""evristic_method()"" или ""RNN()"" -->
                   возвращает кол-во товаров у покупателя на текущем фрейме

        При уходе покупателя:
                   в трекере (vkusmart/tracker/trackers_manager.py) есть список удаленных треков для каждой камеры
                   как только трек попадает в удаленные, его PickCounter также удаляется, перед этим вызвав метод:
                   save_pick_counter_layout -- сохраняет данные обо всем
                   fill_person_all_picks_time -- сохраняет с данной камеры инфу о взятиях/возвратах данного трека в 
                           поле PickCounterStore-- personal_grep(put)_history.
                   Затем, когда человек выходит из магазина(детектится в треккере), из пайплайна дергается метод
                   clusterize_picks -- он кластеризует personal_grep(put)_history, и делает вывод о времени взятия/во-
                           зврата каждого товара. Получаем поля clusterized_grep(put)
                   далее, из общего пайплайна в history записываются листы clusterized_grep(put)
        '''
        print("PICK_COUNTER, release!")
        self.track_id = track_id
        self.shelf_coords = shelf_coords
        self.desicion_maker_type = desicion_maker_type
        self.arm_pos_history = defaultdict(
            lambda: defaultdict(list))  # time ---> left_coords/right_coords ----> (arm bbox)
        self.arm_state_history = defaultdict(
            lambda: defaultdict(int))  # time ---> left_state/right_state ----> (arm_state_result)
        self.arm_class_probs_history = defaultdict(
            lambda: defaultdict(ArmProbs))  # time ---> left_class_probs/right_class_probs ----> (arm_class_probs_result)
        self.arm_velocity_history = defaultdict(
            lambda: defaultdict(Velocity))  # time ---> left_coords/right_coords ----> (class Velocity)
        
        self.arm_distance_history = defaultdict( # AI
            lambda: defaultdict(Distance))  # time ---> left_coords/right_coords ----> (class Distance)

        self.arm_kps_score_history = defaultdict( # AI
            lambda: defaultdict(float))
        
        self.initial_timestamp = timestamp
        self.time_step = time_step

        self.picked = defaultdict(int)
        self.picked[self.initial_timestamp] = 0
        self.put_back = defaultdict(int)
        self.put_back[self.initial_timestamp] = 0
        self.in_basket = defaultdict(int)
        self.in_basket[self.initial_timestamp] = goods_already_picked

        self.pickcounter_layout_dir = pickcounter_layout_dir

#         if desicion_maker_type == 'evristic':
#             self._init_evristic_method(streak, good_in_streak)
        print('desicion_maker_type==', desicion_maker_type)
        if desicion_maker_type == 'evristic':
            self.pick_count_processing = EvristicMethod(streak, good_in_streak, self.initial_timestamp, self.time_step)
        if desicion_maker_type == 'neural_variant_1':
            if path_to_neural_weights == '':
                raise ValueError('No waights for pick_counter inited!')
            self.pick_count_processing = PickNetV1('/mnt/nfs/user6/model.pt', 7, 1)
        
        self.camera_idx = camera_idx

    def _init_evristic_method(self,
                              streak,
                              good_in_streak):
        self.hand_balance = defaultdict(int)  # left_state/right_state ---> balance
        self.hand_balance['left_state'] = 0
        self.hand_balance['right_state'] = 0
        self.streak = streak
        self.good_in_streak = good_in_streak
        self.already_counted_pick = {"left_state": False, "right_state": False}
        self.already_put_back = {"left_state": False, "right_state": False}

    def initial_update(self,
                       cur_time: int,
                       current_arm_pos: defaultdict(list),
                       current_arm_state: defaultdict(int),
                       current_arm_class_probs: defaultdict(list),
                       current_arm_kps_scores: defaultdict(float)):

        self.arm_pos_history[cur_time] = current_arm_pos
        self.arm_state_history[cur_time] = current_arm_state
        self.arm_class_probs_history[cur_time] = current_arm_class_probs
        self.arm_kps_score_history[cur_time] = current_arm_kps_scores

    def calc_crop_center(self, pos):
        '''
        input: pos: np.array -- tlbr coords of right OR left arm
        output:     list -- center of crop
        '''
        if DEBUG: print('pos=', pos)
        return [pos[0] + int((pos[2] - pos[0]) / 2), pos[1] + int((pos[3] - pos[1]) / 2)]

    def calc_velocity(self,
                      prev_pos: defaultdict(list),
                      prev_time: int,
                      cur_pos: defaultdict(list),
                      cur_time: int
                      ):
        velocity = defaultdict(Velocity)
        if DEBUG:
            print()
            print('cur_time, prev_time, track_id', cur_time, prev_time, self.track_id)
            print('BEFORE POSERROR: armstate[cur_frame]=', self.arm_state_history[cur_time].items())
            print('BEFORE POSERROR: armpos[cur_frame]=', self.arm_pos_history[cur_time].items())
            print('BEFORE POSERROR: armstate[prev_time]=', self.arm_state_history[prev_time].items())
            print('BEFORE POSERROR: armpos[prev_time]=', self.arm_pos_history[prev_time].items())

        for arm_type in ['left_coords', 'right_coords']:
            prev_arm_center = self.calc_crop_center(prev_pos[arm_type])
            cur_arm_center = self.calc_crop_center(cur_pos[arm_type])

            cur_vel = Velocity()
            cur_vel.calc_vel(prev_arm_center, cur_arm_center, cur_time - prev_time)

            velocity[arm_type] = cur_vel

        if DEBUG: print("velocity['right_coords'].x_component=", velocity['right_coords'].x_component)
        return velocity
    
    def calc_distances(self, # AI
                      cur_pos: defaultdict(list),
                      cur_time: int,
                      cur_shelf_coords: list,
                      cur_hip_point: list
                      ):
        distances = defaultdict(Distance)
        if DEBUG:
            print('cur_time, track_id', cur_time, self.track_id)
            print('BEFORE POSERROR: armstate[cur_frame]=', self.arm_state_history[cur_time].items())
            print('BEFORE POSERROR: armpos[cur_frame]=', self.arm_pos_history[cur_time].items())
       
        for arm_type in ['left_coords', 'right_coords']:
            cur_arm_center = self.calc_crop_center(cur_pos[arm_type])

            cur_distances = Distance()
            cur_distances.calc_distns(cur_arm_center, cur_shelf_coords, cur_hip_point)

            distances[arm_type] = cur_distances

        if DEBUG: print("distance['right_coords'].x_component=", distances['right_coords'].x_component)
        return distances
                      

    def is_move_from_shelf_to_body(self, cur_time):
        # если рука приближается к полке то расстояние между центром полки и центром руки уменьшается
        # где моя лодка?
        state_to_coord = {'left_state': 'left_coords', 'right_state': 'right_coords'}
        ans = {'left_state': False, 'right_state': False}
        for arm_state_type in ['left_state', 'right_state']:
            arm_coord_type = state_to_coord[arm_state_type]
            dist_before = calc_dist(
                self.calc_crop_center(self.arm_pos_history[cur_time - self.time_step][arm_coord_type]),
                self.calc_crop_center(self.shelf_coords)
            )
            dist_now = calc_dist(
                self.calc_crop_center(self.arm_pos_history[cur_time][arm_coord_type]),
                self.calc_crop_center(self.shelf_coords)
            )
            if dist_before > dist_now:
                ans[arm_state_type] = True
        return ans #dist_before > dist_now

    def fill_linear_move(self, time_start, time_finish):
        # time_start-- последний заполненный кроп. После него всё пусто
        # time_finish -- последний заполняЕМЫЙ кроп
        duration = time_finish - time_start
        for arm_coord in ['left_coords', 'right_coords']:
            begin = self.calc_crop_center(self.arm_pos_history[time_start][arm_coord])
            end = self.calc_crop_center(self.arm_pos_history[time_finish + 1][arm_coord])
            dx = int((end[0] - begin[0]) / duration)
            dy = int((end[1] - begin[1]) / duration)
            delta_move_vector = [dx, dy, dx, dy]
            for time in range(time_start + 1, time_finish + 1):
                newcoord = [0, 0, 0, 0]
                # пасхалочка
                for i in range(4):
                    newcoord[i] = self.arm_pos_history[time_start][arm_coord][i] + delta_move_vector[i]
                self.arm_pos_history[time][arm_coord] = newcoord[:]

    def update(self,
               cur_time: int,
               current_arm_pos: defaultdict(list),
               current_arm_state: defaultdict(int),
               current_arm_class_probs: defaultdict(list),
               current_center_hip_coords: list,
               current_arm_kps_scores: defaultdict(float)):
        if (cur_time - self.initial_timestamp) % self.time_step != 0:
            raise Exception('SOME problems with time STEP (not stamp). see /master/vkusmart/pick_counter.py)')
        
#         print('current_arm_state===', current_arm_state)
        
        if DEBUG: print('cur_time=', cur_time)
        if DEBUG: print('self.initial_timestamp=', self.initial_timestamp)
        if cur_time == self.initial_timestamp:
            self.initial_update(
                cur_time,
                current_arm_pos,
                current_arm_state,
                current_arm_class_probs,
                current_arm_kps_scores
            )
            return self.in_basket[cur_time]

        # бывает такое, что руки не видны => нет кропа, предпологаем равноменое движение. сохраняем время последней обновы
        last_update_time = max(self.arm_pos_history.keys())
        if DEBUG: print('last_update_time', last_update_time)

        # загружем новую.
        self.arm_pos_history[cur_time] = current_arm_pos
        self.arm_state_history[cur_time] = current_arm_state

        # сглаживаем
        if last_update_time != cur_time - self.time_step:
            self.fill_linear_move(last_update_time, cur_time - 1)

        cur_velocity = defaultdict(Velocity)
        cur_velocity = self.calc_velocity(self.arm_pos_history[cur_time - self.time_step],
                                          cur_time - self.time_step,
                                          self.arm_pos_history[cur_time],
                                          cur_time) 
        
        # Аркадий, из  calc_velocity можно просто вытащить distance до полки
        # используя метод класса Velocoty: dist_point_line, 
        # передвая в него коорд центра кропа и shelf_coords
        
        cur_distance = defaultdict(Distance) # AI
        cur_distance = self.calc_distances(self.arm_pos_history[cur_time],
                                           cur_time,
                                           self.shelf_coords,         
                                           current_center_hip_coords)
        
           
        self.arm_velocity_history[cur_time] = cur_velocity
        self.arm_distance_history[cur_time] = cur_distance # AI    
        goods_in_basket = self.calc_pick_count(cur_time)
        # взято товаров. ЭТО ДЛЯ ДЕБАГА. Реальное кол-во взятых товаров считается после выхода человека из магазина
        return goods_in_basket  

    def calc_pick_count(self, cur_time):
        print('self.camera_idx=', self.camera_idx)
#         if IS_HARDCODE:
#             is_picked = {"left_state": 0, "right_state": 0}
#             ALL_SHIFT = 0 #504
            
#             if self.track_id != 8: # and (self.track_id != 3 or cur_time < 1850):# and self.track_id != 5:
#                 # попробуем заодно эвристику
#                 is_picked = {"left_state": 0, "right_state": 0} #self.pick_count_processing.predict(cur_time,
# #                                            self.arm_state_history,
# #                                            self.is_move_from_shelf_to_body(cur_time)
# #                                           )
#             else:
#                 # ага, да, успехов понять через неделю, что это значит    
# #                 print('see track 8!')
#                 if not HARDCODE_INFO.pc_res[str(self.track_id)].get(str(self.camera_idx)) is None:
#                     for arm_type, pick_info in HARDCODE_INFO.pc_res[str(self.track_id)][str(self.camera_idx)].items():
#                         print('pick_info', pick_info)
#                         for picks in pick_info:
#                             print('cur_time+ALL_SHIFT', cur_time+ALL_SHIFT)
#                             print('picks[0]', picks[0])
#                             print('HARDCODE_INFO.cameras_shift[self.camera_idx]', HARDCODE_INFO.cameras_shift[self.camera_idx])
#                             if cur_time == picks[0] - HARDCODE_INFO.cameras_shift[self.camera_idx]: # тут не кастанули к инту
#                                 is_picked[arm_type + '_state'] += picks[1] - picks[2] 
  
            
        if self.desicion_maker_type == 'evristic':
            is_picked = self.pick_count_processing.predict(cur_time,
                                           self.arm_state_history,
                                           self.is_move_from_shelf_to_body(cur_time)
                                          )
        elif self.desicion_maker_type == 'neural_variant_1':
            seq_len = 25
            if cur_time > seq_len/2 + 1:
                is_picked = {"left_state": 0, "right_state": 0}
                # input format: (1, seq_len, num_features), seq_len = 25, num_features = 7
                armstate_res = [1 if self.arm_state_history[cur_time]['right_state']==i else 0 for i in range(4)]
                input_data = [[
                                self.arm_velocity_history[cur_time+i]['right_coords'].module_to_shelf,
                                self.arm_velocity_history[cur_time+i]['right_coords'].module,
                                self.arm_velocity_history[cur_time+i]['right_coords'].y_component
                              ] + armstate_res
                    for i  in range(-1* int(seq_len/2), int(seq_len/2), 1)]
                print(input_data)
                print(type(input_data))
                formatted_input_data = np.dstack(input_data)
                print(formatted_input_data.shape)
                self.pick_count_processing.predict(formatted_input_data)
            else:
                is_picked = {"left_state": 0, "right_state": 0}
            
        else:
            raise ValueError('This: ', self.desicion_maker_type, ', does not defined!')
        print('is_picked', is_picked)
        # is_picked: {"left_state": 0/1/-1, "right_state": 0/1/-1}
#         is_picked = self.pick_count_processing.predict(cur_time,
#                                            self.arm_state_history,
#                                            is_move_from_shelf_to_body(cur_time, arm_state_type)
#                                           )

        for arm_state_type in ['left_state', 'right_state']:
            if is_picked[arm_state_type] == 1:
                self.picked[cur_time] = 1

            if is_picked[arm_state_type] == -1:
                self.put_back[cur_time] = 1
#         if self.camera_idx == 17:
#             print('prev, cur_in, cur_out:', self.in_basket[cur_time - self.time_step], self.picked[cur_time], self.put_back[cur_time])
        self.in_basket[cur_time] = self.in_basket[cur_time - self.time_step] + self.picked[cur_time] - self.put_back[cur_time]

        return self.in_basket[cur_time]
    
    def get_current_goods_in_basket(self, cur_time):
        return self.in_basket[cur_time]

    def get_full_pick_history(self):
        return self.picked, self.put_back, self.picked
       

    def save_pick_counter_layout(
        self,
        cur_time,
        customer_gone_away: bool,
        begin_time: int = 0,
        camera_idx=-1
    ):
        if cur_time % 50 == 0:
            print("NO LAYOUT IN RELEASE VERSION!")
#         KOSTIL_TMP_LAYOUT_STATE = 'right_state'
#         KOSTIL_TMP_LAYOUT_POS = 'right_coords'
#         KOSTIL_TMP_ARM = 'right'
#         arm_types = ['left', 'right']
#         # if срабатывает, если человек недолго находился в зоне видимостри камеры
#         if len(self.arm_velocity_history) < 30: 
#             return 

#         print('#######################################################')
#         print('customer_gone_away, ID= ', self.track_id)
#         print('#######################################################')
        
#         for arm_type in arm_types:
#             csvData = [[
#                 'frame_num', 'camera_idx', 'arm_type', 'arm_state', 'vx', 'vy',
#                 'v', 'module_to_shelf', 'camera_idx', 'distance_to_shelf', 'distance_to_hip'
#             ]]  
#             pos_type = arm_type + '_coords'
#             state_type = arm_type + '_state'
            
#             for time in range(self.initial_timestamp, cur_time - 30):
#                 print(self.arm_velocity_history[time][pos_type])
#                 if not hasattr(self.arm_velocity_history[time][pos_type], 'x_component'):
#                     csvData.append([
#                         time,
#                         arm_type,
#                         self.arm_state_history[time][state_type],
#                         0,
#                         0,
#                         0,
#                         0,
#                         camera_idx,
#                         self.distance_to_shelf, #AI
#                         self.distance_to_hip
#                     ])
#                 else:
#                     csvData.append([
#                         time,
#                         arm_type,
#                         self.arm_state_history[time][state_type],
#                         self.arm_velocity_history[time][pos_type].x_component,
#                         self.arm_velocity_history[time][pos_type].y_component,
#                         self.arm_velocity_history[time][pos_type].module,
#                         self.arm_velocity_history[time][pos_type].module_to_shelf,
#                         camera_idx,
#                         self.arm_distance_history[time][pos_type].distance_to_shelf, # AI 
#                         self.arm_distance_history[time][pos_type].distance_to_hip
#                     ])
#             if customer_gone_away:
#                 # запись по уходу человека из магазина
#                 csv_filename = self.pickcounter_layout_dir+'/'+str(self.track_id) + '_' + arm_type +'.csv'
#                 if not os.path.exists(self.pickcounter_layout_dir):
#                     os.makedirs(self.pickcounter_layout_dir)
#             else:
#                 # запись каждые 500 кадров
#                 csv_filename = self.pickcounter_layout_dir+'/'+str(self.track_id)+ '_' + arm_type+'/'+str(begin_time)+'.csv'
#                 if not os.path.exists(self.pickcounter_layout_dir + '/' + str(self.track_id)+ '_' + arm_type):
#                     os.makedirs(self.pickcounter_layout_dir + '/' + str(self.track_id)+ '_' + arm_type)
# #             if DEBUG: 
#             print('~~~~~~~~~~~~~~~~~~csv_filename~~~~~~~~~~~~~~~~~~', csv_filename)
#             with open(csv_filename, 'w') as csvFile:
#                 writer = csv.writer(csvFile)
#                 writer.writerows(csvData)


class PickCounterStore():
    # как это всё работает, см описание у PickCounter(выше)
    def __init__(
        self,
        shelf_coords,
        pickcounter_layout_dir: str = '',
        pick_counter_path_to_neural_weights: str = '',
        time_step: int = 1,
        desicion_maker_type: str = 'evristic',
        streak: int = 10,
        good_in_streak: int = 2
    ):
        print("PickCounterStore, release!")
        self.pick_counter_store = defaultdict(
            lambda: defaultdict(PickCounter))  # track_id ---> camera idx --> PickCounter
        self.time_step = time_step
        self.desicion_maker_type = desicion_maker_type
        self.streak = streak
        self.good_in_streak = good_in_streak
        self.shelf_coords = shelf_coords
        self.pickcounter_layout_dir = pickcounter_layout_dir
        self.pick_counter_path_to_neural_weights = pick_counter_path_to_neural_weights
        
        # track_id ---> [frame_num, when good grepped, sum amount of grepped goods]
        self.personal_grep_history = defaultdict(dict)
        # track_id ---> [frame_num, when good put, sum amount of put goods]
        self.personal_put_history = defaultdict(dict) 
        
        # тут будут сохраняться центры кластеров. кластер состоит из решений pick_counter-а
        # и имеют вид "взял" и "положил на место", размазанных по времени, собранных с разных камер,
        # с правой и левой руки (считаем что нельзя взять 1 товар двумя руками)
        self.clusterized_grep = defaultdict(list) # track_id ---> № кадров, на которых взяли товар
        self.clusterized_put = defaultdict(list) # track_id ---> № кадров, на которых положили товар
        
        print("self.pick_counter_store INITED!")

    def new_empty_pick_counter(self,
                               track_id,
                               cur_time,
                               goods_already_picked,
                               camera_idx=-1
                               ):
#         if self.desicion_maker_type == 'evristic':
        self.pick_counter_store[track_id][camera_idx] = PickCounter(
            track_id=track_id,
            timestamp=cur_time,
            time_step=self.time_step,
            goods_already_picked=goods_already_picked,
            desicion_maker_type=self.desicion_maker_type,
            streak=self.streak,
            good_in_streak=self.good_in_streak,
            shelf_coords=self.shelf_coords[camera_idx],
            pickcounter_layout_dir=self.pickcounter_layout_dir,
            path_to_neural_weights=self.pick_counter_path_to_neural_weights,
            camera_idx=camera_idx
        )
#         else:
#             raise ValueError('##### Pick Counter.This desicion_maker_type is not implemented! #####')

    def clusterize_picks(self, track_id):
        self.clusterized_grep[track_id] = one_dim_clasterize(self.personal_grep_history[track_id])
        self.clusterized_put[track_id] = one_dim_clasterize(self.personal_put_history[track_id])
        
        
        
    def fill_person_all_picks_time(self, track_id, camera_idx):
        # когда человек уходит с камеры, сохраняем время взятий/выкладываний товаров (нули удалятся)
        self.personal_grep_history[track_id] = dict(Counter(self.personal_grep_history[track_id]) + Counter(self.pick_counter_store[track_id][camera_idx].picked)) # wow, it's actually works
        
        self.personal_put_history[track_id] = dict(Counter(self.personal_put_history[track_id]) + Counter(self.pick_counter_store[track_id][camera_idx].put_back))
        
    def get_cur_arms_state(self,
                           ID,
                           cur_cam_idx,
                           frame_num
                          ):
        return self.pick_counter_store[ID][cur_cam_idx].arm_state_history[frame_num]
    
    def get_cur_arms_coord(self,
                           ID,
                           cur_cam_idx,
                           frame_num
                          ):
        return self.pick_counter_store[ID][cur_cam_idx].arm_pos_history[frame_num]
    
    # будет вызываться из пайплайна
    def update_all_cameras(self,
                           bb_ids_kps_arms: list,
                           cur_time: int,
                           goods_already_picked: defaultdict(lambda: defaultdict(int)),
                           all_deleted_tracks: defaultdict(list),
                           shelf_ordinal_to_cam_index: list,
                           update_rate = 100
                          ):
        '''
        # как это всё работает, см описание у PickCounter(выше)
        # тут небольшое уточнение
        Новый инстанс pick_counter-а будет появлятся при возникновения нового track_id в переданном словаре(bb_ids_kps_arms)
        Удаление... Удаление будет только при подтвеждения удаления от трекера.
        Отсутствие track_id в bb_ids_kps_arms не означает отсутствие трека. Можно смотреть на её отсутствие в течение
        max_age - 20, но малейшее изменение в логике--- и все рушится

        goods_already_picked def_dict(int): track_id ---> amount_goods_picked
        (предыдущий кадр. В PickCounter есть история взятий, но есть "новые" треки, которые уже что-то взяли
        и инициализация нулем не подходит)
        bb_ids_kps_arms:
        {
            'track_id': int,

            'bbox': Tuple(4),  # tlbr

            'keypoints_info': {
                'keypoints': tensor.Size(17, 2),
                'kp_score': tensor.Size(17, 1),
                'proposal_score': tensor.Size(1)  # float
            },

            'crops_info': {
                'left_crop': types.Image,
                'rigth_crop': types.Image,
                'left_coords': Tuple(4),  # tlbr
                'right_coords':  Tuple(4),  # tlbr
                'left_state': int,
                'right_state': int
            }
        }
        '''
        if shelf_ordinal_to_cam_index==[]:
            print('===No shelf cameras?===')
            return {}
        
        goods_picked = defaultdict(lambda: defaultdict(int))  # track_id ---> camera_idx ---> the_amount_of_goods_picked

        # ищем удаленные треки. (человек ушел с камеры)
        for camera_idx, del_tracks_ids in all_deleted_tracks.items():
            for del_id in del_tracks_ids:
                if self.pick_counter_store.get(del_id.track_id) is None or self.pick_counter_store[del_id.track_id].get(camera_idx) is None:
                    continue
                # для camera_idx не явл-ся shelf, не будет даже создан экземпляр
                # т.к. экз-ры создаются в этом методе(ниже), а все camera_idx исп-ся там-- из bb_ids_kps_arms.
                # _
                # сохраняем скорость, результаты arm_state и т.д. для устаревшего трека
                self.pick_counter_store[del_id.track_id][camera_idx].save_pick_counter_layout(
                    cur_time, customer_gone_away=True, camera_idx=camera_idx
                )
                # сохраняем только время взятий (для последующей оффлайн-кластеризации)
                self.fill_person_all_picks_time(del_id.track_id, camera_idx)
                
                del self.pick_counter_store[del_id.track_id][camera_idx]
                

        for camera_idx, one_img_bb_ids_kps_arms in enumerate(bb_ids_kps_arms):
            camera_idx=shelf_ordinal_to_cam_index[camera_idx] # cause we devide shelf/hall 
            for one_person_info in one_img_bb_ids_kps_arms:
                ID = one_person_info['track_id']
                
                if not self.pick_counter_store[ID]:
                    self.pick_counter_store[ID] = defaultdict(PickCounter)

                if self.pick_counter_store[ID].get(camera_idx) is None:  # проверка, без создания инстанса
                    if goods_already_picked.get(ID) is None:
                        goods_already_picked[ID] = defaultdict(int)
                    if goods_already_picked[ID].get(camera_idx) is None:
                        goods_already_picked[ID][camera_idx] = 0
                    if DEBUG: print('CREATE NEW! CUR_TIME = ', cur_time)
                    self.new_empty_pick_counter(
                        ID,
                        cur_time,
                        goods_already_picked[ID][camera_idx],
                        camera_idx
                    )

                if len(self.pick_counter_store[ID][camera_idx].arm_state_history) % update_rate == 0:
                    self.pick_counter_store[ID][camera_idx].save_pick_counter_layout(
                        cur_time, customer_gone_away=False, 
                        begin_time=cur_time-499, camera_idx=camera_idx
                    )
                    
                current_arm_pos = {'left_coords': one_person_info['crops_info']['left_coords'],
                                   'right_coords': one_person_info['crops_info']['right_coords']}
                current_arm_state = {'left_state': one_person_info['crops_info']['left_state'],
                                     'right_state': one_person_info['crops_info']['right_state']}
                current_arm_class_probs = {'left_class_probs': one_person_info['crops_info']['left_class_probs'],
                                   'right_class_probs': one_person_info['crops_info']['right_class_probs']}
                if DEBUG:
                    print("#######################")
                    print("ID:", ID)
                    print('current_arm_pos: ', current_arm_pos.items())
                    print('current_arm_state', current_arm_state.items())
                    print("#######################")

                # Get keypoints for person AI
                current_kps_one_person = one_person_info['keypoints_info']['keypoints']
                current_left_hip_coords = current_kps_one_person[11]
                current_right_hip_coords = current_kps_one_person[12]
                current_center_hip_coords = np.array([
                    float(current_left_hip_coords[0] + ((current_left_hip_coords[0] - current_right_hip_coords[0])/2.)),
                    float(current_left_hip_coords[1] + ((current_left_hip_coords[1] - current_right_hip_coords[1])/2.))
                ])
                
                current_arm_kps_scores = {'left_kps_score': float(one_person_info['keypoints_info']['kp_score'].numpy()[11]),
                                          'right_kps_score': float(one_person_info['keypoints_info']['kp_score'].numpy()[12])}
                    
                goods_picked[ID][camera_idx] = self.pick_counter_store[ID][camera_idx].update(
                    cur_time,
                    current_arm_pos,
                    current_arm_state,
                    current_arm_class_probs,
                    current_center_hip_coords,
                    current_arm_kps_scores)

        return goods_picked
    