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
    X_train, y_train = preprocesing.prepare_data_train(h_parameters)

    X_train = np.array(X_train)
    X_train = np.reshape(X_train, (len(X_train), h_parameters.image_size, h_parameters.image_size, 1))
    y_train = np.array(y_train, dtype=object)

    cv.imshow('img1', X_train[0])
    cv.waitKey(0)
    cv.imshow('img2', X_train[3])
    cv.waitKey(0)
    cv.imshow('img3', X_train[8])
    cv.waitKey(0)

    print('y train -- ', y_train[5])
    print('y train -- ', y_train[9])
    print('y train -- ', y_train[19])

    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)

    # If the test data exists, reshape it and normalize it.
    if h_parameters.with_test:

        # Load the test data.
        X_test, y_test = preprocesing.prepare_data_test(h_parameters)

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (len(X_test), h_parameters.image_size, h_parameters.image_size, 1))
        y_test = np.array(y_test, dtype=object)

        print("y_test: ", y_test.shape)
        print("X_test: ", X_test.shape)

        cv.imshow('test_img1', X_test[0])
        cv.waitKey(0)
        cv.imshow('test_img2', X_test[3])
        cv.waitKey(0)
        cv.imshow('test_img3', X_test[8])
        cv.waitKey(0)

        print('y test -- ', y_test[5])
        print('y test -- ', y_test[9])
        print('y test -- ', y_test[19])
    else:
        X_test = None
        y_test = None


    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    h_parameters = parameters.get_config()
    loading_data(h_parameters)

