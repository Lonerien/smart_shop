from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

import math
import cv2


@contextmanager
def open_video(video_path: Path, start_frame_number=0, mode='r', *args):
    '''
    Context manager to work with cv2 videos
    Modes are either 'r' for read or 'w' write
    which returns cv2.VideoCapture or cv2.VideoWriter respectively.
    Additional arguments passed according to OpenCV documentation
    '''
    if video_path is None:
        yield None
    else:
        if mode == 'r':
            video = cv2.VideoCapture(video_path.as_posix(), *args)
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            
        elif mode == 'w':
            video = cv2.VideoWriter(video_path.as_posix(), *args)
        else:
            raise ValueError('Incorrect open mode "{}", "r" or "w" expected!'.format(mode))
        if not video.isOpened(): raise ValueError('Video {} is not opened!'.format(video_path))
        try:
            yield video
        finally:
            video.release()

            
def frames(video: cv2.VideoCapture):
    '''
    Generator of frames of the video provided
    '''
    while True:
        retval, frame = video.read()
        if not retval:
            break
        # this is to return RGB image instead of BGR
        frame = frame[..., ::-1]
        yield frame

        
def read_frames(video_path: Path, start_frame_number: int=0, verbose = False):
    '''
    Combines functions open_video and frames for compactness
    '''
    if verbose: print('start_frame_number=', start_frame_number)
    with open_video(video_path, start_frame_number) as video_cap:
        yield from frames(video_cap)
        
# тут генератор, который вызывает генератор, вызываемый генератором. (экзибит.пнг)
# Влад был слишком хорош в написании кода
# TODO: понять и переписать
def path_lol(path_lol):
    while True:
        yield str(path_lol)
        
def return_path(video_path: Path):
    yield from path_lol(video_path)

        
def convert_bbox(bbox: tuple, fr: str, to: str) -> tuple:
    '''
    Converts bounding box from one fromat to other
    Available formats:
        * 'xywh' - top left point (xz, y), width and height
        * 'tlbr' - top left point (x, y) and bottom right point (x, y)
    Note: make enum for `fr`, `to`
    '''
    if fr == 'xywh' and to == 'tlbr':
        x, y, w, h = bbox
        return [ x, y, x + w, y + h ]
    elif fr == 'tlbr' and to == 'xywh':
        l, t, r, b = bbox
        return [l, t, r - l, b - t]

    raise NotImplementedError('Sorry, this functionality is not currently available')
    
    
def draw_boxes(image: np.array, bboxes: list, ids: list=None, box_type: str='detector'):
    '''
    Draw the bounding boxes on the image, return image with boxes
    
    :param image: `np.array` with shape (H, W, 3) in RGB format
    :param bboxes: list of `np.array`s, each np.array contains coordinates of BB in tlbr format (see `convert_bbox`)
    :param ids: if not None, put ID from 'ids' on top of each bounding box
    
    :return image: image with the drawn boxes
    '''
    
    if box_type == 'detector':
        ids = range(len(bboxes))  # just in bboxes order
        bboxes_for_draw = bboxes
    if box_type == 'tracker':
        ids = [tr_id for bbox, tr_id in bboxes]
        bboxes_for_draw = [bbox for bbox, tr_id in bboxes]
    if box_type == 'vis_to_file':
        ids = ids
        bboxes_for_draw = bboxes
    
#     print('bboxes=', bboxes)
#     print('bboxes_for_draw=', bboxes_for_draw)
    digit_height = 35
    digit_width = 23
    
    for ID, bbox in zip(ids, bboxes_for_draw):
#         print('ID, bbox, box_type==', ID, bbox, box_type)
        image = cv2.rectangle(
            image, 
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (255,255,255),
            2,
        )
        
        
        if box_type == 'detector':
            note = 'det=' + str(ID)
            note_coords = (int(bbox[0]), int(bbox[1])+digit_height)
            note_color = (255,0,0)
        elif box_type == 'tracker' or 'vis_to_file':
            note = str(ID)#'tr_id=' + str(ID)
            note_coords = (int(bbox[0]), int(bbox[1])+digit_height)
            note_color = (0,0,0)
        else:
            raise ValueError
            
        image = cv2.rectangle(
            image, 
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[0]) + digit_width * len(str(note)), int(bbox[1])+digit_height),
            (255,255,255),
            cv2.FILLED
        ) 
            
        image = cv2.putText(
            image,
            text=note,
            org=note_coords,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=note_color,
            thickness=2, 
            lineType=cv2.LINE_AA,
        )
    return image if type(image) == type(np.array([])) else image.get()

def draw_pick_counter(image: np.array, bboxes_for_draw: list, ids: list=None, box_type: str='detector', picked={}, draw_in_frame={}, cam_id=-999):
    
    for ID, bbox in zip(ids, bboxes_for_draw):
#         print('ID, bbox, box_type==', ID, bbox, box_type)
        box_right_corner_x = int(bbox[2])
        box_right_corner_y = int(bbox[1])
        digit_height = 35
        digit_width = 23
        
        note_color = (0,255,0)
        
        note = '0'
        for cur_t_id, cur_cam_idx, cur_gp in draw_in_frame:
            if cur_cam_idx == '27': print('27 camera')
            if ID == int(cur_t_id) and cam_id == int(cur_cam_idx):
                note = str(cur_gp)
#         if cur_cam_idx == '27': print('27 camera, note=', note)
        
        # закоммент, т.к. сейчас реализуется не столь хардкодно
#         for info in picked:
#             tr_id, good_picked = info[0], info[1]
#             if tr_id == ID:
#                 note = str(good_picked)
#                 break
                
        note_coords = (box_right_corner_x - digit_width*len(str(note)), box_right_corner_y+digit_height-5)
        
        image = cv2.rectangle(
            image, 
            (box_right_corner_x - digit_width * len(str(note)), box_right_corner_y),
            (box_right_corner_x, box_right_corner_y+digit_height),
            (255,255,255),
            cv2.FILLED
        )        
        
        image = cv2.putText(
            image,
            text=note,
            org=note_coords,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=note_color,
            thickness=2, 
            lineType=cv2.LINE_AA
        )
    return image


def draw_poses(image: np.array, keypoints_infos: list):
    '''
    Draw the keypoints of persons on the image. Returns image with drawn keypoints
    
    :param image: `np.array` with shape (H, W, 3) in RGB format
    :param keypoints_infos: list of dicts:
        {
            'keypoints': tensor.Size(14, 2),  # in COCO format !
            'kp_score': tensor.Size(14, 1), 
            'proposal_score': tensor.Size(1)
        }
    
    :return image: image with the drawn keypoints
    '''
    # COCO2017 format: 17 body parts
    coco_part_names = {
        0: 'Nose', 1: 'LEye', 2: 'REye', 3: 'LEar', 4: 'REar',  # head
        5: 'LShoulder', 6: 'RShoulder', 7: 'LElbow', 8: 'RElbow', 9: 'LWrist', 10: 'RWrist',  # body
        11: 'LHip', 12: 'RHip', 13: 'LKnee', 14: 'RKnee', 15: 'LAnkle', 16: 'RAnkle'  # legs
    }
    pose_graph = np.asarray([
        [0,1], [0,2], [1,3], [2,4],  # head
        [5,6], [5,7], [7,9], [6,8], [8,10],  # body and arms
        [11,12], [11,13], [13,15], [12,14], [14,16],  # legs
        [6,12], [5,11]  # join body and legs
    ])
    line_colors = 4 * [(255, 0, 0)] + 5 * [(255, 255, 0)] + 5 * [(0, 0, 255)] + 2 * [(255, 255, 0)]
    
    for keypoints_info in keypoints_infos:
        coords = keypoints_info['keypoints'].numpy()
        scores = keypoints_info['kp_score'].numpy()
        # draw points
        for kpoint_coords, kpoint_score in zip(coords, scores):
            image = cv2.circle(
                img=image,
                center=(kpoint_coords[0], kpoint_coords[1]), 
                radius=6, 
                color=(0, 255, 0),
                thickness=-1
            )
        # draw the lines of skeleton
        for idx, pair in enumerate(pose_graph):
            cv2.line(
                img=image,
                pt1=(coords[pair[0]][0], coords[pair[0]][1]),
                pt2=(coords[pair[1]][0], coords[pair[1]][1]),
                color=line_colors[idx],
                thickness=3
            )
            
    return image


def draw_crops(image: np.array, crops_infos: list):
    '''
    Draw the crops of person's arms on the image. Returns image with drawn arm crop boxes
    
    :param image: `np.array` with shape (H, W, 3) in RGB format
    :param crops_infos: list of dicts:
        {
            'left_coords': tensor.Size(4),
            'right_coords': tensor.Size(4), 
            'left_state': tensor.Size(1)
            'right_state': tensor.Size(1)
        }
    
    :return image: image with the drawn arm crop boxes
    '''
    # 0="empty", 1="with an item", 2="trash", 3="bag/pocket"
    STATE_COLORS = {0: (255, 0, 0), 1: (0, 255, 0), 2: (255, 255,0), 3: (0,0,255)}
    for crop_info in crops_infos:
        for arm_type in ['left', 'right']:
#             image = cv2.rectangle(
#                 image, 
#                 (crop_info[f'{arm_type}_coords'][0], crop_info[f'{arm_type}_coords'][1]),
#                 (crop_info[f'{arm_type}_coords'][2], crop_info[f'{arm_type}_coords'][3]),
#                 STATE_COLORS[crop_info[f'{arm_type}_state']],
#                 3,
#             )
            center_x = int(crop_info[f'{arm_type}_coords'][0] + (crop_info[f'{arm_type}_coords'][2] - crop_info[f'{arm_type}_coords'][0])/2)
            center_y = int(crop_info[f'{arm_type}_coords'][1] + (crop_info[f'{arm_type}_coords'][3] - crop_info[f'{arm_type}_coords'][1])/2)
            image = cv2.circle(image, (center_x, center_y), 75, STATE_COLORS[crop_info[f'{arm_type}_state']], 3)
            
    return image


def get_video_info(video_path: Path, *args):
    '''
    Get all the parameters of the video by its video_path
    '''
    video = cv2.VideoCapture(video_path.as_posix(), *args) 
    video_info = {}
    video_info['path'] = video_path.resolve().as_posix()
    video_info['fourcc'] = int(video.get(cv2.CAP_PROP_FOURCC))
    video_info['fps'] = video.get(cv2.CAP_PROP_FPS)
    video_info['height'] = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    video_info['width'] = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video.release()
    return video_info

def rotate_img(image, angle):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angle, 1)

    rad = math.radians(angle)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg
    
def crop_img(img, tlbr):
    return img[tlbr[1]:tlbr[3], tlbr[0]:tlbr[2]]


def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def one_dim_clasterize(x):
    '''
    input: array like this [1,1,5,6,1,5,10,22,23,23,50,51,51,52,100,112,130,500,512,600,12000,12230]
    then, get clusters: 
        cluster 0: [ 1  1  5  6  1  5 10 22 23 23 50 51 51 52]
        cluster 1: [100 112 130]
        cluster 2: [500 512]
        cluster 3: [12000]
        cluster 4: [12230]
        cluster 5: [600]
    then, culculate avg element of each cluster
    output: middle of clusters
    '''
    print(x)
    print(type(x))
    print(x.values())
    q = 1
    for val in x.values():
        q+=1
        print('val=', val)
        if  val == []:
            return []                                                                                     
    if q == 1:
        return []
    x = x.values()
    a = list(zip(x,np.zeros(len(x))))
#     print(a)
    X = np.array(a, dtype=np.int)
    bandwidth = estimate_bandwidth(X, quantile=0.1)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    cluster_centers = []
    
    for k in range(n_clusters_):
        my_members = labels == k
        print("cluster {0}: {1}".format(k, X[my_members, 0]))
        cluster_centers.append(np.average(X[my_members, 0]))
    cluster_centers= list(map(lambda x: int(x), cluster_centers))
    return cluster_centers    
        