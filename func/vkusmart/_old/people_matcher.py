from collections import defaultdict
from .types import BB_FVs

import numpy as np
from scipy.spatial.distance import cosine


def dist_metric_euc(previous_features_of_person, cur_features):
    a = previous_features_of_person
    b = cur_features
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b)


def dist_metric_cos(previous_features_of_person, cur_features):
    a = previous_features_of_person
    b = cur_features
    a, b = np.asarray(a), np.asarray(b)
    return cosine(a, b)


def identify(
    frame, 
    gallery, 
    timestamp, 
    images, 
    bb_fvs, 
    dist_metric='euc', 
    neighs_num=10, 
    verbose=False
):
    '''Assigns id to detected bounding boxes
    '''
    matching_result = []
    if verbose: i = 0
    for cam_id, (img, bb_fv) in enumerate(zip(images, bb_fvs)):
        if verbose:
            i += 1
            print('identify(): cam_id', i, 'num people:', len(bb_fv))
        for bb, query_fv in bb_fv:
            weights = []
            min_distance = 10000
            closest_person = -1
            for person, fv in gallery.features():
                if dist_metric == 'euc':
                    cur_best_distance = dist_metric_euc(fv, query_fv)
                    if verbose:
                        print('person', person, 'fv', fv, 'cur_best_distance', cur_best_distance)
                    if cur_best_distance < min_distance:
                        closest_person = person
                        min_distance = cur_best_distance

                elif dist_metric == 'cos':
                    cur_best_distance = dist_metric_cos(fv, query_fv)
                    if cur_best_distance < min_distance:
                        closest_person = person
                        min_distance = cur_best_distance

                elif dist_metric == 'knn':
                    cur_weight = 1. / dist_metric_euc(fv, query_fv)
                    weights.append({'weight': cur_weight, 'pers_id': person})

                    weights = sorted(weights, key=lambda k: k['weight'], reverse=True)[:neighs_num]

                    reduced = defaultdict(int)
                    for el in weights:
                        reduced[el['pers_id']] += el['weight']

                    weights = [{'pers_id': pers_id, 'weight': dist} for pers_id, dist in reduced.items()]
                    cur_w = 0

                    for el in weights:
                        if el['weight'] > cur_w:
                            cur_w = el['weight']
                            closest_person = el['pers_id']

                else:
                    raise ValueError("This dist metric doesn't exist!")

            matching_result.append({
                'person_id': closest_person, 
                'cam_id': cam_id,
                'timestamp': timestamp,
                'frame': frame,
                'bbox': bb
            })

    if verbose:        
        print(len(matching_result))

    return matching_result
