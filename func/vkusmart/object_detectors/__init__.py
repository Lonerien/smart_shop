#from __future__ import absolute_import

# from . import yolov3_tf
from . import yolov3_pytorch


__factory = {
#     'yolov3_tf': yolov3_tf.YOLOv3,
    'yolov3_pytorch': yolov3_pytorch.YOLOv3
#     'ssd': ...
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """
    Create a detector instance.
    Parameters
    ----------
    name : str
        Model name. Can be one of 'yolov3_tf', 'yolov3_pytorch'.
    """
    if name not in __factory:
        raise KeyError("Unknown model:", name)
    return __factory[name](*args, **kwargs)