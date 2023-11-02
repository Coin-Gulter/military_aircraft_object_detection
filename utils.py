import cv2 as cv
import numpy as np

def format_image(img, image_size):
    # Function to resize and format an input image to a specified image size.

    # Get the height and width of the input image.
    height, width, = img.shape 
    # Calculate the maximum dimension (height or width).
    max_size = max(height, width)
    # Calculate the ratio by which the image will be resized to fit the specified image size.
    ratio = max_size / image_size
    # Calculate the new width and height after resizing.
    new_width = int(width / ratio)
    new_height = int(height / ratio)
    # Create a new image of the specified size filled with zeros.
    new_size = (new_width, new_height)
    resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
    new_image = np.zeros((image_size, image_size), dtype=np.uint8)
    # Copy the resized image into the top-left corner of the new image.
    new_image[0:new_height, 0:new_width] = resized

    return new_image

def get_key_from_dict(dict, value):
    # Function to retrieve a key from a dictionary based on its corresponding value.

    # Extract the keys and values from the dictionary.
    keys = list(dict.keys())
    items = list(dict.values())

    # Find the index of the specified value in the list of values.
    index = items.index(value)
    # Return the key corresponding to the found index.
    return keys[index]
