import preprocesing
import cv2 as cv
import parameters
import numpy as np


def loading_data(h_parameters):
    """
    Load the data from the disk and prepare it for training and testing.

    Args:
        h_parameters (dict): Hyperparameters of the model.

    Returns:
        X_train, y_train, X_test, y_test: Training and test data.
    """

    print('Start data loader...')

    # Load the training data.
    X_train, y_train_obj_exist, y_train_classification, y_train_detection = preprocesing.prepare_data_train(h_parameters)

    X_train = np.array(X_train)/255
    X_train = np.reshape(X_train, (len(X_train), h_parameters.image_size, h_parameters.image_size, 1))
  
    y_train_obj_exist = np.array(y_train_obj_exist, dtype=np.float16)
    y_train_classification = np.array(y_train_classification, dtype=np.float16)
    y_train_detection = np.array(y_train_detection, dtype=np.float16)/h_parameters.image_size

    # cv.imshow('img1', X_train[0])
    # cv.waitKey(0)
    # cv.imshow('img2', X_train[3])
    # cv.waitKey(0)
    # cv.imshow('img3', X_train[8])
    # cv.waitKey(0)

    # print('y train classification -- ', y_train_classification[0])
    # print('y train classification -- ', y_train_classification[3])
    # print('y train classification -- ', y_train_classification[8])

    # print('y train detection -- ', y_train_detection[0])
    # print('y train detection -- ', y_train_detection[3])
    # print('y train detection -- ', y_train_detection[8])

    print("X_train: ", X_train.shape)
    print("y_train objects exist: ", y_train_obj_exist.shape)
    print("y_train classification: ", y_train_classification.shape)
    print("y_train detection: ", y_train_detection.shape)

    # If the test data exists, reshape it and normalize it.
    if h_parameters.with_test:

        # Load the test data.
        X_test, y_test_obj_exist, y_test_classification, y_test_detection = preprocesing.prepare_data_test(h_parameters)

        X_test = np.array(X_test)/255
        X_test = np.reshape(X_test, (len(X_test), h_parameters.image_size, h_parameters.image_size, 1))
    
        y_test_obj_exist = np.array(y_test_obj_exist, dtype=np.float16)
        y_test_classification = np.array(y_test_classification, dtype=np.float16)
        y_test_detection = np.array(y_test_detection, dtype=np.float16)/h_parameters.image_size

        # cv.imshow('test_img1', X_test[0])
        # cv.waitKey(0)
        # cv.imshow('test_img2', X_test[3])
        # cv.waitKey(0)
        # cv.imshow('test_img3', X_test[8])
        # cv.waitKey(0)

        # print('y test classification -- ', y_test_classification[0])
        # print('y test classification -- ', y_test_classification[3])
        # print('y test classification -- ', y_test_classification[8])

        # print('y test detection -- ', y_test_detection[0])
        # print('y test detection -- ', y_test_detection[3])
        # print('y test detection -- ', y_test_detection[8])

        print("X_test: ", X_test.shape)
        print("y_test objects exist: ", y_test_obj_exist.shape)
        print("y_test classification: ", y_test_classification.shape)
        print("y_test detection: ", y_test_detection.shape)

    else:
        X_test = None
        y_test_classification = None
        y_test_detection = None
        y_test_obj_exist = None

    return X_train, {'yolo_output_0' : y_train_obj_exist, 'yolo_output_1' : y_train_classification, 'yolo_output_2' : y_train_detection }, X_test, {'yolo_output_0' : y_test_obj_exist, 'yolo_output_1' : y_test_classification, 'yolo_output_2' : y_test_detection }


if __name__ == "__main__":
    h_parameters = parameters.get_config()
    loading_data(h_parameters)

