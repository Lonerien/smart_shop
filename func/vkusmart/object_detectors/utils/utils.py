# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_names(file_name):
    names = {}
    with open(file_name) as f:
        for id, name in enumerate(f):
            names[id] = name
    return names


def drawrect(drawcontext, xy, outline, width):
    x1, y1, x2, y2 = xy
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)
    
    
def convert_to_original_size(box, size, original_size):
    ratio = original_size / size
    box = box.reshape(2, 2) * ratio
    return list(box.reshape(-1))


def draw_boxes(boxes, img, cls_names, detection_size):
    draw = ImageDraw.Draw(img)
    COLORS = {class_id: tuple(np.random.randint(low=0, high=255, size=3)) for class_id in cls_names.keys()}
#     FONT_FILE = '.fonts/ibmplexsans/IBMPlexSans-Regular.ttf'
    FONT = ImageFont.load_default()  # ImageFont.truetype(FONT_FILE, 15)
    WIDTH = 2
    
    for cls, bboxs in boxes.items():
        color = COLORS[cls]
        for box, score in bboxs:
            box = convert_to_original_size(box, np.array(detection_size), np.array(img.size))
            drawrect(draw, box, outline=color, width=WIDTH)
            draw.text(box[:2], '{}'.format(cls_names[cls]), fill=color, font=FONT)