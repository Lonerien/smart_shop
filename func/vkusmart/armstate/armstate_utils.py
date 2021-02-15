import math


def f(elbow, wrist, normal_arm_length=100, min_size=150, max_size=150):
    ''' Returns coordinates of crop. With min_size=50 it will be rectangle
    '''
    ex, ey = elbow
    wx, wy = wrist
    cx = wx + (wx - ex) / 4
    cy = wy + (wy - ey) / 4
    a = math.atan2(wy - ey, wx - ex)
    # d = sin (angle between arm and camera axis)
    d = abs((wx - ex) + (wy - ey) * 1j) / normal_arm_length
    w = min_size + (max_size - min_size) * abs(math.cos(a)) * d
    h = min_size + (max_size - min_size) * abs(math.sin(a)) * d
    return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))


def clip(p, a, b):
    return max(a, min(b, p))


def crop(img, x, y, w, h):
    ih = img.shape[0]
    iw = img.shape[1]
    return img[clip(y, 0, ih):clip(y+h, 0, ih), clip(x, 0, iw):clip(x+w, 0, iw)]

