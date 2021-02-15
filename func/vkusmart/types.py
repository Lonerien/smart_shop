from typing import Union, List, Tuple, Any, Dict, DefaultDict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np


Image = np.ndarray
Images = List[np.ndarray]

Timestamp = Union[datetime, int]

Directory = Union[Path, str]
Filename = Union[Path, str]

BoundingBox = Tuple[float]
BoundingBoxes = List[BoundingBox]

Score = Union[np.float, float]
BoundingBoxesWithScores = List[Tuple[BoundingBox, Score]]

FeatureVector = np.ndarray
FeatureVectors = List[FeatureVector]

BB_FV = Tuple[BoundingBox, FeatureVector]
BB_FVs = List[BB_FV]

Keypoints = np.ndarray

BB_KP = Tuple[BoundingBox, Keypoints]
BB_KPs = List[BB_KP]

# places is dict describing one place person detected in
# it contains fields: cam_id, timestamp, frame_number, bbox
Place = Dict[str, Any]
Places = List[Place]

''' in tracker '''
# Bounding boxes, stored in tracker
Track_BB = DefaultDict[int, list]

# pick_history consists of GoodsPicks.
# GoodsPicks: list of pairs (time, camera) -- add new, when person take good
GoodsPicks = List[DefaultDict[int, int]]

# arm_crops: list[dict]  time --> dict{camera, {'right_arm_crop': Image, 'left_arm_cop': Image}}
# Needed for testing armstate and new armstate dataset collection
ArmCrops = DefaultDict[int, DefaultDict[str, Image]]
ArmCropsHistory = List[ArmCrops]

# arm_coords: list[dict]  time --> dict{camera, BBox_KeyPoints} 
# Needed for testing armstate
ArmCoord = DefaultDict[str, list] # left/right ---> coords

# arm state:
ArmState = DefaultDict[str, int] # left/right ---> state

# trajectory: dict[dict]  time --> dict{camera, BBox} -- history of person locations
Position = Tuple[int, List[int]]
MulticamPosition = DefaultDict[int, List[int]]
Trajectory = DefaultDict[int, MulticamPosition]

# camera_id: dict int --> int
# from "number issued when filling" ----> "real_id"
# old camera number === real_id
# т.е. было all_trackers[cam_id: int] ----> стало all_trackers[cam_id[some_key]: DefDict[int]  ]
CamsId = DefaultDict[int, int]