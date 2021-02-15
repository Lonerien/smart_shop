from typing import Tuple

from .types import Filename
from .registry import Registry
from .gallery import Gallery


def get_prefilled(
    prefill_file: Filename,
) -> Tuple[Registry, Gallery]:
    '''Prefills given registry and gallery form prefill_file

    For now this funct just drops gallery, but it may use existing values as well
    In fact this should be `load` method of joined Registry/Gallery class...

    Args:
        prefill_file: file .npy with dumped gallery
    '''
    gallery = Gallery.load(prefill_file)
    registry = Registry()
    for person in gallery.persons:
        registry.create(person)
    return registry, gallery
