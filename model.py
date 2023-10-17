import numpy as np
import cv2 as cv
import os
import data_loader
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dropout, Dense, Flatten, MaxPooling2D



class page_ai():
    """
    Class for the page_ai model.

    Args:
        h_parameters (dict): Hyperparameters for the model.
    """

    def __init__(self, h_parameters):
        """
        Constructor for the page_ai class.

        Args:
            h_parameters (dict): Hyperparameters for the model.
        """

        self.h = h_parameters
        self.input_size = (self.h.image_size, self.h.image_size, 1)

        # Check if the pretrained model is used.
        if self.h.pretrained:
            print('Loading pretrained model...')
            self.model = tf.keras.models.load_model(os.path.join(self.h.save_model_folder, self.h.save_model_name))
            print('Model loaded')
        else:
            # Create the model.
            print('Creating model...')
            inputs = tf.keras.layers.Input(shape=self.input_size)
                    
            feature_extractor = self.build_feature_extractor(inputs)

            model_adaptor = self.build_model_adaptor(feature_extractor)

            classification_head = self.build_classifier_head(model_adaptor)

            regressor_head = self.build_regressor_head(model_adaptor)

            self.model = tf.keras.Model(inputs = inputs, outputs = [classification_head, regressor_head])
            print('Model created')

        # Summarize the model.
        self.model.summary()

    def build_feature_extractor(self, inputs):

        x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=self.input_size)(inputs)
        x = tf.keras.layers.AveragePooling2D(2,2)(x)

        x = tf.keras.layers.Conv2D(32, kernel_size=3, activation = 'relu')(x)
        x = tf.keras.layers.AveragePooling2D(2,2)(x)

        x = tf.keras.layers.Conv2D(64, kernel_size=3, activation = 'relu')(x)
        x = tf.keras.layers.AveragePooling2D(2,2)(x)

        return x

    def build_model_adaptor(self, inputs):
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        return x

    def build_classifier_head(self, inputs):
        return tf.keras.layers.Dense(self.h.num_classes, activation='softmax', name = 'classifier_head')(inputs)

    def build_regressor_head(self, inputs):
        return tf.keras.layers.Dense(units = self.h.regresion_number, name = 'regressor_head')(inputs)

    def train(self):
        """
        Function to train the page_ai model.

        Args:
            None.

        Returns:
            None.
        """

        # Load the training and test data.
        X_train, y_train, X_test, y_test = data_loader.loading_data(self.h)

        # Compile the model.
        self.model.compile(optimizer=self.h.optimizer, loss=self.h.loss[0], metrics=self.h.metrics[0])

        # Train the model.
        self.model.fit(X_train, 
                        y_train, 
                        epochs=self.h.num_epochs, 
                        batch_size=self.h.batch_size, 
                        validation_split=self.h.validation_split)

        # Evaluate the model on the test set.
        if self.h.with_test:
            test_loss, test_acc = self.model.evaluate(np.array(X_test),
                                                        np.array(y_test))
            print('Tested lost:', test_loss)
            print('Tested accuracy:', test_acc)

        model_name = self.h.save_model_name

        # Save the model.
        self.model.save(os.path.join(self.h.save_model_folder, model_name))
    

    def predict(self, img:np.ndarray):
        """
        Function to predict the rotation of an image.

        Args:
            img (np.ndarray): Image to be predicted.

        Returns:
            int: Predicted rotation angle.
        """

        # Check if the image is grayscale or RGB.
        if len(img.shape) < 3:
            pass
        elif len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            raise Exception(f'Given image not in correct format - {img.shape} , \
                                try using image of shape (x, y, 1) for grayscale or (x, y, 3) if image colorful')

        # Preprocess the image.
        img_size = (self.h.image_size, self.h.image_size)

        # Resize the image.
        resized_image = cv.resize(img, img_size)

        # Normalize the image.
        binary_image = resized_image.astype('float32') / 255

        # Reshape the image.
        reshape_image = binary_image.reshape((1, self.h.image_size, self.h.image_size, 1))

        # Make the prediction.
        result = self.model.predict(reshape_image)

        return result

