from typing import List

from .types import Timestamp, Images, BB_FVs
from .registry import Registry
from .gallery import Gallery

from .people_matcher import identify


class Resolver(object):
    def __init__(self, gallery: Gallery, registry: Registry):
        self.gallery = gallery
        self.registry = registry

    def resolve(
        self,
        frame: int,
        timestamp: Timestamp,
        images: Images,
        bb_fvs: List[BB_FVs],
        dist_metric: str='euc'
    ) -> None:
        '''Resolves all seen detections

        This processes places updates, exit & enter events, tills presence

        Args:
            frame: number of frame as of :py:funct:`.providers.VideoProvider.frames`
            timestamp: see :py:funct:`.providers.VideoProvider.frames`
            images: see :py:funct:`.providers.VideoProvider.frames`
            bb_fvs: bounding boxes and corresponding featurevectors
                see :py:funct:`.reids.Reid.extract`
        '''
        # update registry with new `person_id` <-> (cam_id, timestamp, frame, bbox) entries
        new_entries = self._identify(frame, timestamp, images, bb_fvs, dist_metric)
        for entry in new_entries:
            self.registry.get(entry['person_id']).places.append(
                {k:entry[k] for k in entry if k != 'person_id'}
            )
        # TODO exit enter check
        self._exit_enter_check(frame, timestamp, images, bb_fvs)
        # TODO till check
        self._till_check(frame, timestamp, images, bb_fvs)

    def _identify(self, frame, timestamp, images, bb_fvs, dist_metric):
        '''Assigns ids to detected bounding boxes
        '''
        return identify(frame, self.gallery, timestamp, images, bb_fvs, dist_metric=dist_metric)

    def _exit_enter_check(self, frame, timestamp, images, bb_fvs):
        '''Process enter and exit logic

        In this function creading and deleting of persons occures
        '''
        pass

    def _till_check(self, frame, timestamp, images, bb_fvs):
        '''Checks tills precense, updates registry accordingly
        '''
        pass
