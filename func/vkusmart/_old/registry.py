from typing import Dict, List
import pickle
from pathlib import Path

from .types import Timestamp, Filename, Places


class RegistryEntry(object):
    '''Represents information of one person's positions

    Usage is to set attributes as necessary
    In case of places you want to call .append() new place which is dict
    '''
    def __init__(
        self,
        person: int,
        places: Places=None,
        time_enter: Timestamp=None,
        time_exit: Timestamp=None,
        time_till_enter: Timestamp=None,
        time_till_exit: Timestamp=None,
        till_visited: bool=False,
    ):
        '''
        Args:
            person: person's identity
            places: list of person's places, see :py:class:`.types.Place`
            time_enter: time of enterance to the shop
            time_exit: time of exit from the shop
            time_till_enter: time of enterance to tills zone
            time_till_exit: time of exit from tills zone
            till_visited: if this person had visited tills
        '''
        self.person = person
        self.places = places if places is not None else []
        self.time_enter = time_enter
        self.time_exit = time_exit
        self.time_till_enter = time_till_enter
        self.time_till_exit = time_till_exit
        self.till_visited = till_visited

    def __str__(self):
        return (
            'RegistryEntry of:\n'
            f'person: {self.person}\n'
            f'places len: {len(self.places)}\n'
            f'time_enter: {self.time_enter}\n'
            f'time_exit: {self.time_exit}\n'
            f'time_till_enter: {self.time_till_enter}\n'
            f'time_till_exit: {self.time_till_exit}\n'
            f'till_visited: {self.till_visited}\n'
        )


class Registry(object):
    '''Sotres and provides interface for persons information

    TODO write __getitem__ instead of get method or inherit dict/defaultdict
    '''
    def __init__(self):
        '''
        Entries maps person ids to their RegistryEntry
        '''
        self._entries: Dict[int, RegistryEntry] = {}

    def __str__(self):
        return f'Registry with {self._entries.keys()} persons'

    def create(self, person: int) -> None:
        '''Creates new Entry in Registry
        '''
        self._entries[person] = RegistryEntry(person)

    def get(self, person: int):
        '''Returns registry of the person

        This method is used for both reading and updating Entries
        '''
        return self._entries[person]

    def dump(self, filename: Filename, *, method: str='.pickle') -> None:
        '''Saves current registry state to given 
        '''
        filename = Path(filename).with_suffix(method)
        with open(filename, 'wb') as file:
            pickle.dump(self._entries, file)

    @classmethod
    def load(cls, filename: Filename):
        registry = cls()
        with open(filename, 'rb') as file:
            entries = pickle.load(file)
        registry._entries = entries
        return registry
