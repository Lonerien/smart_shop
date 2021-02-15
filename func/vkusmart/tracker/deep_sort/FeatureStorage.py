from collections import defaultdict
from . import nn_matching 
import numpy as np
import time


'''
Feature_store -- есть у каждого трека
У каждого трека есть "время жизни без детекций"-- max_age
Логика:
Хотим хранить 5 фичей за последние 15 кадров 
Реализация по структурам данных:
features : defaultdict(list)  track_id --> [[feture vector], [feture vector], ...]
    внутри: кольцевой буфер на основе list: list + pointer (timestamp)
timestamp: defaultdict(int) 

'''
class Feature_store(object):
    def __init__(self, capacity=10, update_time=5):
        self.tracks_features = defaultdict(list)
        self._timestamps = defaultdict()
        self._capacity = 10#capacity
        self.update_time = 5#update_time     #хз, мб это внешняя логика
        
    def add_new_feature(self, track_id, feature_vector):
        if self.tracks_features[track_id] == []:
            self._timestamps[track_id] = 0
            self.tracks_features[track_id] = [ [] for i in range(self._capacity)]
            
        self.tracks_features[track_id][self._timestamps[track_id]] = feature_vector
#         print('and here-------->', type(self.tracks_features[track_id][self._timestamps[track_id]]))
        self._timestamps[track_id] = (self._timestamps[track_id] + 1) % 5
        
#         print('tracks_features[track_id]', self.tracks_features[track_id])
        
    def get_features_by_id(self, track_id):
        return self.tracks_features[track_id]
    
    def add_several_FV(self, track_id, track_features):  # используется для создание локальной const копии.
#         print('now here----------->', type(track_features[0]))
        self.tracks_features[track_id] = track_features
    
        
    def most_similar_track_id(self, feature_vector, N=5):
        '''
        kkn: 
            вход: 1 фич-вектор, N 
            по всем персонам:
                считаем расстояния от всех фич-вект до входного фич-вектора
                выбираем N наименьших
                суммируем 
                если получившаяся сумма минимальна, то сохраняем track_id персоны
        '''
        #TODO optimize!
        min_dist = 1000000
        huge_val = 9999999
        best_match = -1
        all_matches = []
        for track_id, track_features in self.tracks_features.items():
#             print('track_id in storage:', track_id)
            relevance_dist = []
            for feature in track_features:
                if feature != []:
#                     print('----------->', type(feature))
                    prob_relevance_matrix = nn_matching._cosine_distance([feature_vector], feature)[0]
                    relevance_dist.append(prob_relevance_matrix.reshape(-1)[0])
            if relevance_dist != []:
                N_ = min(N, len(relevance_dist))
                distance = np.sum(np.sort(relevance_dist)[:N_]) / float(N_)
#                 print(distance)
                #time.sleep(5)
                all_matches.append((distance, track_id))
                if min_dist > distance:
                    min_dist = distance
                    best_match = track_id
        return best_match, all_matches
    
# fs = Feature_store()
# fs.add_new_feature(1, [3., 1.05, 0.95])
# fs.add_new_feature(1, [3., 1.01, 0.95])
# fs.add_new_feature(1, [3., 1.1, 0.95])
# fs.add_new_feature(1, [3., 1.9, 0.95])
# fs.add_new_feature(1, [3., 1.9, 0.95])
# fs.add_new_feature(1, [1., 1.9, 0.95])
# fs.add_new_feature(1, [1., 1.9, 0.95])
# fs.add_new_feature(1, [3., 1.9, 0.95])
# fs.add_new_feature(1, [3., 1.9, 0.95])
# fs.add_new_feature(1, [3., 1.9, 0.95])

# fs.add_new_feature(2, [2., 1.05, 0.95])
# fs.add_new_feature(2, [2., 1.01, 0.95])
# fs.add_new_feature(2, [2., 1.1, 0.95])

# print(fs.most_similar_track_id([1., 1., 1.], N=2))