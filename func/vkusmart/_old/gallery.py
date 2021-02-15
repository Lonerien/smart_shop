from typing import Dict, List, Iterable, Tuple

import numpy as np

from .types import FeatureVector, FeatureVectors, Filename


class GalleryEntry(object):
    '''Stores featurevectors of one person
    '''
    def __init__(self, person: int) -> None:
        '''
        Args:
            id_: person's identity

        Object is created with empty features
        '''
        self.person = person
        self._features: FeatureVectors = []

    def __str__(self):
        return f'GalleryEntry of {self.person} with {len(self._features)} featurevectors'

    def add(self, features: FeatureVector) -> None:
        '''Adds new featurevector to this person
        '''
        self._features.append(features)

    def features(self) -> Iterable[FeatureVector]:
        '''Iterates over all stored featurevectors of this person
        '''
        yield from self._features


class Gallery(object):
    '''Stores featurevectors of persons

    Note: init doesn't have any params
    Dimentionality of given featurevectors must be the same!

    This class supports both creating new empty person (then id is returned)
        and adding featurevector for any person (it get created if it don't exist)

    In fact Gallery and Registry is one entity cause it manages persons idenitites
        but they were splitted for some reason. Convenience is that Gallery now
        is leading entity, so in create function it returns id unlike Registry

    _gallery maps persons to their GalleryEntries
    '''
    def __init__(self):
        self._gallery: Dict[int, GalleryEntry] = {}

    def create(self) -> int:
        '''Creates new person in gallery with given featurevector

        Returns:
            id of newly created person
        '''
        person = max(self._gallery.keys(), default=-1) + 1
        self._gallery[person] = GalleryEntry(person)
        return person

    def add(self, person: int, features: FeatureVector) -> None:
        '''Adds new featurevector for given camera to person
        '''
        # defaultdict by hand =)
        if person not in self.persons:
            self._gallery[person] = GalleryEntry(person)
        self._gallery[person].add(features)

    def features(self) -> Iterable[Tuple[int, FeatureVector]]:
        '''Iterates over all stored featurevectors with their persons

        Yeilds:
            (person, features) - person id and one featurevector of this person
        '''
        for person, entry in self._gallery.items():
            for features in entry.features():
                yield person, features

    @property
    def persons(self): # -> dict_keys # I don't know how to access this type
        '''Returns view on all person ids in gallery
        '''
        return self._gallery.keys()

    def delete(self, person: int) -> None:
        '''Removes person from gallery
        '''
        self._gallery.pop(person)

    def dump(self, filename: Filename, *, method: str='.npy') -> None:
        '''Dumps current gallery state to given file .npy
        '''
        fv_dim = next(self.features())[1].shape
        gallery_dtype = np.dtype([
            ('person', np.int, ()),
            ('featurevector', np.float32, fv_dim),
        ])

        gallery_dump = np.array(
            list(self.features()),
            dtype=gallery_dtype,
        )
        np.save(filename.with_suffix(method), gallery_dump)

    @classmethod
    def load(cls, filename: Filename): # -> Gallery # python 3.6 doesn't allow to return self class =(
        gallery = cls()
        loaded = np.load(filename)
        for item in loaded:
            gallery.add(item['person'], item['featurevector'])
        return gallery
