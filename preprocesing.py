import xml.etree.ElementTree as et
import cv2 as cv
import time
import re
import numpy as np
import time
import parameters

input_size = 244

class preprocesing():
    """
    Class for preprocessing the data.

    Args:
        h_parameters (dict): Hyperparameters of the model.
    """

    def __init__(self, h_parameters):
        self.h = h_parameters

        # Get the paths to the training and test images.
        with open(self.h.train_file, mode='r') as tr_fl:
            files_name = tr_fl.readlines()
        self.tr_files_tree = [et.parse(self.h.data_annotation_path + re.sub('\n','',elem) +'.xml') for elem in files_name]

        self.tr_roots = [tree.getroot() for tree in self.tr_files_tree]
        self.train_files = [root.find('filename').text for root in self.tr_roots]
        self.train_file_path = [self.h.train_img_path+file_name for file_name in self.train_files]
        self.train_file_path = list(map(self.file_jpg_filter, self.train_file_path))


        if self.h.with_test:
            with open(self.h.test_file, mode='r') as ts_fl:
                files_name = ts_fl.readlines()
            self.ts_files_tree = [et.parse(self.h.data_annotation_path + re.sub('\n','',elem) +'.xml') for elem in files_name]

            self.ts_roots = [tree.getroot() for tree in self.ts_files_tree]
            self.test_files = [root.find('filename').text for root in self.ts_roots]
            self.test_file_path = [self.h.test_img_path+file_path for file_path in self.test_files]
            self.test_file_path = list(map(self.file_jpg_filter, self.test_file_path))

    def file_jpg_filter(self, filename):
        if re.search(r'\.[a-z]{3,4}$', filename).group(0) == '.jpg':
            filename = re.sub(' ', '', filename)
            return filename
        else:
            filename = re.sub(' ', '', filename)
            return re.sub(r'\.[a-z]{3,4}$', '.jpg', filename)

    def encode_in_one_hot(self,number):
        img_classes = self.h.image_classes[0]
        onehot = [0]*len(img_classes)
        onehot[number] = 1
        return onehot

    def format_image(self, img, bbox):
        height, width, = img.shape 
        max_size = max(height, width)
        r = max_size / self.h.image_size
        new_width = int(width / r)
        new_height = int(height / r)
        new_size = (new_width, new_height)
        resized = cv.resize(img, new_size, interpolation= cv.INTER_LINEAR)
        new_image = np.zeros((self.h.image_size, self.h.image_size), dtype=np.uint8)
        new_image[0:new_height, 0:new_width] = resized

        new_box = []
        for box in bbox:
            x, y, w, h = box[0], box[1], box[2], box[3]
            # box = [int((x - 0.5*w)* width / r), int((y - 0.5*h) * height / r), int(w*width / r), int(h*height / r)]
            box = [int(x / r), int(y / r), int(w / r), int(h / r)]
            new_box.extend([(box[0]+box[2]/2)/self.h.image_size, (box[1]+box[3]/2)/self.h.image_size,
                            (box[2]/2)/self.h.image_size, (box[3]/2)/self.h.image_size])

        return new_image, new_box
    
    def download_img(self, file_path, bbox):
        img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        # print('img -- ', img)
        new_img, new_bbox = self.format_image(img, bbox)
        return new_img, new_bbox



def prepare_data_train(h_parameters):
    """
    Prepare the training data.

    Args:
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        X_train, y_train: Training data.
    """

    preprocess = preprocesing(h_parameters)

    X_train = []
    y_train = []

    length = len(preprocess.train_file_path)
    percentage = 0
    prev_time = time.time()

    for index, img_path in enumerate(preprocess.train_file_path):

        if int((index*100)/length) != percentage:
            percentage = int((index*100)/length)
            eta = time.strftime("%H:%M:%S", time.gmtime((100 - percentage) * (time.time() - prev_time)))

            print(f'Percentage of generated train images = {percentage}% -- ETA = {eta}')

            prev_time = time.time()

        bbox = []
        labels = []

        for obj in preprocess.tr_roots[index].findall('object'):
            label_number = preprocess.h.image_classes[0][obj.find('name').text]
            labels.extend(preprocess.encode_in_one_hot(label_number))
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            w, h = (xmax-xmin),(ymax-ymin)
            bbox.append([xmin, ymin, w, h])

        
        # Read the image and augment it.
        img, bbox = preprocess.download_img(img_path, bbox)
        X_train.append(img)
        # print('labels -- ', labels, 'bbox -- ', bbox)
        number_of_missing_bbox_values = (4*preprocess.h.num_objects-len(bbox))
        number_of_missing_label_values = (20*preprocess.h.num_classes-len(labels))

        labels.extend([0]*number_of_missing_label_values)
        bbox.extend([0]*number_of_missing_bbox_values)
        y_train.append([labels, bbox])

    return X_train, y_train


def prepare_data_test(h_parameters):
    """
    Prepare the test data.

    Args:
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        X_test, y_test: Test data.
    """

    preprocess = preprocesing(h_parameters)

    X_test = []
    y_test = []

    length = len(preprocess.test_file_path)
    percentage = 0
    prev_time = time.time()

    for index, img_path in enumerate(preprocess.test_file_path):

        if int((index*100)/length) != percentage:
            percentage = int((index*100)/length)
            eta = time.strftime("%H:%M:%S", time.gmtime((100 - percentage) * (time.time() - prev_time)))

            print(f'Percentage of generated test images = {percentage}% -- ETA = {eta}')

            prev_time = time.time()

        bbox = []
        labels = []

        for obj in preprocess.ts_roots[index].findall('object'):
            label_number = preprocess.h.image_classes[0][obj.find('name').text]
            labels.extend(preprocess.encode_in_one_hot(label_number))
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            w, h = (xmax-xmin),(ymax-ymin)
            bbox.append([xmin, ymin, w, h])

        
        # Read the image and augment it.
        img, bbox = preprocess.download_img(img_path, bbox)
        X_test.append(img)
        # print('labels -- ', labels, 'bbox -- ', bbox)
        number_of_missing_bbox_values = (4*preprocess.h.num_objects-len(bbox))
        number_of_missing_label_values = (20*preprocess.h.num_classes-len(labels))

        labels.extend([0]*number_of_missing_label_values)
        bbox.extend([0]*number_of_missing_bbox_values)
        y_test.append([labels, bbox])

    return X_test, y_test


if __name__ == '__main__':
    h_parameters = parameters.get_config()
    xtrain, ytrain = prepare_data_train(h_parameters)
    print('xtrain -- ',xtrain)
    print('ytrain -- ', ytrain)

    xtest, ytest = prepare_data_test(h_parameters)
    print('xtrain -- ',xtrain)
    print('ytrain -- ', ytrain)