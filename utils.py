import cv2 as cv
import numpy as np

def format_image(img, image_size):
    height, width, = img.shape 
    max_size = max(height, width)
    ratio = max_size / image_size
    new_width = int(width / ratio)
    new_height = int(height / ratio)
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((image_size, image_size), dtype=np.uint8)
    new_image[0:new_height, 0:new_width] = resized

    return new_image

def get_key_from_dict(dict, value):
    keys = list(dict.keys())
    items = list(dict.values())
    index = items.index(value)
    return keys[index]
