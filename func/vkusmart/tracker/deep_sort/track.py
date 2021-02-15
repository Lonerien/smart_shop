# vim: expandtab:ts=4:sw=4
import numpy as np
from collections import defaultdict
# from .FeatureStore import Feature_store

'''from vkusmart.pick_counter import PickCounter'''

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None, start_time=-1, camera_type='hall'):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age
        print('self._max_age=', self._max_age)        
        self.start_frame = -1
        self.last_seen_frame = -1
        
        self.begin_position = -1
        self.num_frame_begin = -1  #тоже самое, что и start_frame, но используется в другой логической части, пока оставлю
        self.change_env_frame = -1
        self.walk_history = np.array([])
        
        self.cassa_duration = 0
        self.visit_cassa = False
        
        self.start_time = start_time  # enter shop
        self.finish_time = -1         # go away from shop

#         self.feature_store = defaultdict(Feature_store)
        
#         if camera_type=='shelf':
#             self.pick_counter = PickCounter(
#                  timestamp = start_time,
#                  time_step = 1,
#                  goods_already_picked = 0,
#                  current_arm_pos: ,
#                  current_arm_state: ,
#                  shelf_coords: shelf[N],
#                  desicion_maker_type: str='evristic',
#                  streak:int=12,
#                  good_in_streak:int=10)
        
    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        #print('add feature:')
        #print(len(detection.feature))
        self.features.append(detection.feature)
        
        #print('from track: len(features) == ', len(self.features))

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
        if self.state == TrackState.Tentative:
            return self.features
        return None
    
    def special_update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        
        # self.features.append(detection.feature) # фичи были добавлены в special_initiate_track
        # хз почему, но их нет. TODO ---^
        # self.features = detection.feature # но так не работает .......
        #print('from track: len(features) == ', len(self.features))

        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.Confirmed
        print('call special')

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
    
    def is_move_to_street(self, boxs_history: list, decicion_making_time: int):
        """ Returns True if person gos out of shop"""
        def calc_center(bbox: list):
            # box: tlbr
            return [int((bbox[2]+bbox[0])/2), int((bbox[3]+bbox[1])/2)]
        
        def is_go_out_one_step(bb1, bb2):   
            print('------------------------------------')
#             print('bb1, bb2=', bb1, bb2)
#             print('bb2[1]-bb1[1]=', bb2[1]-bb1[1])
            
            bb1, bb2 = calc_center(bb1), calc_center(bb2)
            
#             print('centers: bb1, bb2=', bb1, bb2)
#             print('centers: bb2[1]-bb1[1]=', bb2[1]-bb1[1])
            
            if bb2[1]-bb1[1] < 0:
                return 1
            return 0
        
        is_move_out = []
        for time in range(1, len(boxs_history) - decicion_making_time):
            print('time=', time)
            is_move_out.append( is_go_out_one_step(boxs_history[time-1], boxs_history[time]))
        
        print('sum(is_move_out),  0.6 * len(is_move_out)', sum(is_move_out), 0.6 * len(is_move_out))
        if sum(is_move_out) > 0.8 * len(is_move_out):
            return True
        return False
