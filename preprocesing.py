import xml.etree.ElementTree as et
import cv2 as cv
import time
import re
import random
import numpy as np
import time
import parameters
import keras.utils as utils


class preprocesing():

    def __init__(self, h_parameters):
        # Constructor for the preprocessing class, responsible for data preparation.
        self.h = h_parameters

        # Read the list of training file names from a text file.
        with open(self.h.train_file, mode='r') as tr_fl:
            files_name = tr_fl.readlines()

        # Parse the XML files corresponding to the training data.
        self.tr_files_tree = [et.parse(self.h.data_annotation_path + re.sub('\n','',elem) +'.xml') for elem in files_name]


        # Limit the number of training images if specified in the hyperparameters.
        if self.h.number_train_images < len(self.tr_files_tree):
            self.tr_files_tree = self.tr_files_tree[:self.h.number_train_images]

        # Shuffle the list of training data for randomness.
        random.shuffle(self.tr_files_tree)

        # Extract the root elements of the parsed XML files.
        self.tr_roots = [tree.getroot() for tree in self.tr_files_tree]

        # Extract training file names and build their file paths.
        self.train_files = [root.find('filename').text for root in self.tr_roots]
        self.train_file_path = [self.h.train_img_path+file_name for file_name in self.train_files]
        self.train_file_path = list(map(self.file_jpg_filter, self.train_file_path))


        if self.h.with_test:
            # If testing is enabled, repeat the same process for test data.

            # Read the list of test file names from a text file.
            with open(self.h.test_file, mode='r') as ts_fl:
                files_name = ts_fl.readlines()

            # Parse the XML files corresponding to the test data.
            self.ts_files_tree = [et.parse(self.h.data_annotation_path + re.sub('\n','',elem) +'.xml') for elem in files_name]

            # Limit the number of test images if specified in the hyperparameters.
            if self.h.number_test_images < len(self.ts_files_tree):
                self.ts_files_tree = self.ts_files_tree[:self.h.number_test_images]

            # Shuffle the list of test data for randomness.
            random.shuffle(self.ts_files_tree)

            # Extract the root elements of the parsed XML files.
            self.ts_roots = [tree.getroot() for tree in self.ts_files_tree]
            # Extract test file names and build their file paths.
            self.test_files = [root.find('filename').text for root in self.ts_roots]
            self.test_file_path = [self.h.test_img_path+file_path for file_path in self.test_files]
            self.test_file_path = list(map(self.file_jpg_filter, self.test_file_path))

    def file_jpg_filter(self, filename):
        # Function to ensure that file names have the '.jpg' extension.

        # Check if the file name already has the '.jpg' extension.
        if re.search(r'\.[a-z]{3,4}$', filename).group(0) == '.jpg':
            filename = re.sub(' ', '', filename)
            return filename
        else:
            # If not, replace the extension with '.jpg'.
            filename = re.sub(' ', '', filename)
            return re.sub(r'\.[a-z]{3,4}$', '.jpg', filename)

    def check_obj_grid_part(self, obj_bbox):
        # Function to determine the grid part (cell) of an object's bounding box.

        # Calculate the size of each grid cell.
        grid_size = self.h.image_size / self.h.grid_partition
        
        # Calculate the grid cell in the x-direction.
        grid_x = int(obj_bbox[0] / grid_size)

        # Ensure that the grid_x value is within bounds.
        if grid_x >= self.h.grid_partition:
            grid_x = self.h.grid_partition-1

        # Calculate the grid cell in the y-direction.
        grid_y = int(obj_bbox[1] / grid_size)

        # Ensure that the grid_y value is within bounds.
        if grid_y >= self.h.grid_partition:
            grid_y = self.h.grid_partition-1

        return (int(grid_x), int(grid_y))

    def format_bbox(self, bbox, ratio):
        # Function to format a bounding box with respect to a given ratio.
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        # Calculate the new bounding box coordinates and dimensions.
        box = [int(x / ratio), int(y / ratio), int(w / ratio), int(h / ratio)]
        # Convert the bounding box into YOLO format (center x, center y, width, height).
        new_bbox = np.array([(box[0]+box[2]/2), (box[1]+box[3]/2), (box[2]/2), (box[3]/2)])
        return new_bbox

    def format_image(self, img):
        # Function to resize and format an input image to a specified image size.

        # Get the height and width of the input image.
        height, width, = img.shape 

        # Calculate the maximum dimension (height or width).
        max_size = max(height, width)
        # Calculate the ratio by which the image will be resized to fit the specified image size.
        ratio = max_size / self.h.image_size
        # Calculate the new width and height after resizing.
        new_width = int(width / ratio)
        new_height = int(height / ratio)
        # Create a new image of the specified size filled with zeros.
        new_size = (new_width, new_height)
        resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
        # Create a new image of the specified size filled with zeros.
        new_image = np.zeros((self.h.image_size, self.h.image_size), dtype=np.uint8)
        # Create a new image of the specified size filled with zeros.
        new_image[0:new_height, 0:new_width] = resized

        return new_image, ratio



def prepare_data_train(h_parameters):
    # Function to prepare training data.

    # Create a preprocessing object to handle data preparation.
    preprocess = preprocesing(h_parameters)

    # Initialize lists to store training data and labels.
    X_train = []
    y_train_classification = []
    y_train_detection = []
    y_train_obj_exist = []


    # Get the number of training images.
    length = len(preprocess.train_file_path)
    percentage = 0
    prev_time = time.time()

    # Loop through training images and annotations.
    for index, img_path in enumerate(preprocess.train_file_path):

        # Update progress percentage and estimated time remaining.
        if int((index*100)/length) != percentage:
            percentage = int((index*100)/length)
            eta = time.strftime("%H:%M:%S", time.gmtime((100 - percentage) * (time.time() - prev_time)))

            print(f'Percentage of generated train images = {percentage}% -- ETA = {eta}')

            prev_time = time.time()

        # Read the image in grayscale.
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        # Resize the image and calculate the resizing ratio.
        img, ratio = preprocess.format_image(img)
        X_train.append(img)

        # Initialize arrays for bounding boxes, labels, and object existence.
        bboxes = np.zeros((h_parameters.grid_partition, h_parameters.grid_partition, h_parameters.bbox_number*h_parameters.objects_number))
        labels = np.zeros((h_parameters.grid_partition, h_parameters.grid_partition, h_parameters.num_classes*h_parameters.objects_number))
        obj_exist = np.zeros((h_parameters.grid_partition, h_parameters.grid_partition, h_parameters.objects_number+1))

        # Initialize an array to keep track of the number of objects in each grid cell.
        objects_numbers = np.zeros((h_parameters.grid_partition, h_parameters.grid_partition))
        
        for obj in preprocess.tr_roots[index].findall('object'):

            label_number = preprocess.h.image_classes[0][obj.find('name').text]
            label = utils.to_categorical(label_number, h_parameters.num_classes)
            
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            w, h = (xmax-xmin),(ymax-ymin)
            bbox =  preprocess.format_bbox([xmin, ymin, w, h], ratio)

            grid_x, grid_y = preprocess.check_obj_grid_part(bbox)

            obj_num = int(objects_numbers[grid_x, grid_y])

            start_bbox_point =  int(h_parameters.bbox_number*obj_num)
            end_bbox_point = start_bbox_point + h_parameters.bbox_number

            start_label_point =  int(h_parameters.num_classes*obj_num)
            end_label_point = start_label_point + h_parameters.num_classes

            bboxes[grid_x, grid_y, start_bbox_point:end_bbox_point] = bbox
            labels[grid_x, grid_y, start_label_point:end_label_point] = label
            obj_exist[grid_x, grid_y, obj_num + 1] = 1

            if obj_num < (h_parameters.objects_number-1):
                objects_numbers[grid_x, grid_y] += 1

        y_train_classification.append(labels)
        y_train_detection.append(bboxes)
        y_train_obj_exist.append(obj_exist)

    return X_train, y_train_obj_exist, y_train_classification, y_train_detection


def prepare_data_test(h_parameters):
    preprocess = preprocesing(h_parameters)

    X_test = []
    y_test_classification = []
    y_test_detection = []
    y_test_obj_exist = []

    length = len(preprocess.test_file_path)
    percentage = 0
    prev_time = time.time()

    for index, img_path in enumerate(preprocess.test_file_path):

        if int((index*100)/length) != percentage:
            percentage = int((index*100)/length)
            eta = time.strftime("%H:%M:%S", time.gmtime((100 - percentage) * (time.time() - prev_time)))

            print(f'Percentage of generated test images = {percentage}% -- ETA = {eta}')

            prev_time = time.time()

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        img, ratio = preprocess.format_image(img)
        X_test.append(img)

        bboxes = np.zeros((h_parameters.grid_partition, h_parameters.grid_partition, h_parameters.bbox_number*h_parameters.objects_number))
        labels = np.zeros((h_parameters.grid_partition, h_parameters.grid_partition, h_parameters.num_classes*h_parameters.objects_number))
        obj_exist = np.zeros((h_parameters.grid_partition, h_parameters.grid_partition, h_parameters.objects_number))

        objects_numbers = np.zeros((h_parameters.grid_partition, h_parameters.grid_partition))
        
        for obj in preprocess.tr_roots[index].findall('object'):

            label_number = preprocess.h.image_classes[0][obj.find('name').text]
            label = utils.to_categorical(label_number, h_parameters.num_classes)
            
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            w, h = (xmax-xmin),(ymax-ymin)
            bbox =  preprocess.format_bbox([xmin, ymin, w, h], ratio)

            grid_x, grid_y = preprocess.check_obj_grid_part(bbox)

            obj_num = int(objects_numbers[grid_x, grid_y])

            start_bbox_point =  int(h_parameters.bbox_number*obj_num)
            end_bbox_point = start_bbox_point + h_parameters.bbox_number

            start_label_point =  int(h_parameters.num_classes*obj_num)
            end_label_point = start_label_point + h_parameters.num_classes

            bboxes[grid_x, grid_y, start_bbox_point:end_bbox_point] = bbox
            labels[grid_x, grid_y, start_label_point:end_label_point] = label
            obj_exist[grid_x, grid_y, obj_num] = 1

            if obj_num < (h_parameters.objects_number-1):
                objects_numbers[grid_x, grid_y] += 1

        y_test_classification.append(labels)
        y_test_detection.append(bboxes)
        y_test_obj_exist.append(obj_exist)

    return X_test, y_test_obj_exist, y_test_classification, y_test_detection, 


if __name__ == '__main__':
    h_parameters = parameters.get_config()
    xtrain, ytrain_obj_existens, ytrain_classification, ytrain_detection = prepare_data_train(h_parameters)
    # print('xtrain -- ',xtrain)
    # print('ytrain classification -- ', ytrain_classification)
    # print('ytrain detection -- ', ytrain_detection)
    # print('ytrain obj existens -- ', ytrain_obj_existens)

    xtest, ytest_obj_existens, ytest_classification, ytest_detection = prepare_data_test(h_parameters)
    # print('xtest -- ',xtest)
    # print('ytest classification -- ', ytest_classification)
    # print('ytest detection -- ', ytest_detection)
    # print('ytest objects existens -- ', ytest_obj_existens)
