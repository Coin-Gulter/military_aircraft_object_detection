import numpy as np
import cv2 as cv
import os
import data_loader
import utils
import parameters
from itertools import repeat
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, MaxPooling2D, UpSampling2D, ZeroPadding2D, LeakyReLU, Add, Concatenate, Lambda, Input, Softmax
from tensorflow.keras.regularizers import l2

 
class BatchNormalization(tf.keras.layers.BatchNormalization):
    def call(self, x, training=False):
        if training is None: training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class bbox_ai():
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

            self.model = self.yolo_v3(self.h.image_size, self.h.objects_number, self.h.num_classes, self.h.bbox_number)

            # self.model = tf.keras.Model(inputs = inputs, outputs = [classification_head, regressor_head])
            print('Model created')

        # Summarize the model.
        self.model.summary()

    def backbone_conv(self, x, filters, size, strides=1, batch_norm=True):
        if strides == 1:
            padding = 'same'
        else:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)
            padding = 'valid'
        x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding, 
                   use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x
    
    def backbone_res(self, x, filters):
        previous = x
        x = self.backbone_conv(x, filters, 2, 1)
        x = self.backbone_conv(x, filters, 3)
        x = Add()([previous , x])
        return x
    
    def backbone_block(self, x, filters, blocks):
        x = self.backbone_conv(x, filters, 3, strides=2)
        for _ in repeat(None, blocks):
            x = self.backbone_res(x, filters)
        return x
    
    def darknet(self, name=None):
        x = inputs = Input([self.h.image_size, self.h.image_size, self.h.input_channels])
        x = self.backbone_conv(x, 16, 3)
        x = self.backbone_block(x, 32, 1)
        x = self.backbone_block(x, 64, 2)
        x = x_36 = self.backbone_block(x, 128, 4)
        x = x_61 = self.backbone_block(x, 256, 4)
        x = self.backbone_block(x, 512, 2)
        return tf.keras.Model(inputs, (x_36, x_61, x), name=name)
    
    def yolo_conv(self, filters, name=None):
        def yolo_conv(x_in):
            if isinstance(x_in, tuple):
                inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
                x, x_skip = inputs

                x = self.backbone_conv(x, filters, 1)
                x = UpSampling2D(2)(x)
                x = Concatenate()([x, x_skip])
            else:
                x = inputs = Input(x_in.shape[1:])

            x = self.backbone_conv(x, filters, 1)
            x = self.backbone_conv(x, filters * 2, 3)
            x = self.backbone_conv(x, filters, 1)
            x = self.backbone_conv(x, filters * 2, 3)
            x = self.backbone_conv(x, filters, 1)
            return Model(inputs, x, name=name)(x_in)
        return yolo_conv

    def yolo_output(self, filters, classes, name=None, softmax_activation=False):
        def yolo_output(x_in):
            x = inputs = Input(x_in.shape[1:])
            x = self.backbone_conv(x, filters * 2, 3)
            x = self.backbone_conv(x, classes, 1, batch_norm=False)
            x = Dense(filters*2)(x)
            x = Dense(self.h.grid_partition*self.h.grid_partition*classes)(x)
            x = Lambda(lambda x: tf.reshape(x, (-1, self.h.grid_partition, self.h.grid_partition, classes)))(x)
            if softmax_activation:
                x = Softmax(axis=-1)(x)
            return tf.keras.Model(inputs, x, name=name)(x_in)
        return yolo_output
    
    def yolo_v3(self, size, objects_classes, classification_classes, bbox_classes):
            x = inputs = Input([size, size, self.h.input_channels])

            x_36, x_61, x = self.darknet(name='yolo_darknet')(x)

            x = self.yolo_conv(256, name='yolo_conv_0')(x)
            output_0 = self.yolo_output(512, objects_classes, name='yolo_output_0', softmax_activation=True)(x)

            x = self.yolo_conv(128, name='yolo_conv_1')((x, x_61))
            output_1 = self.yolo_output(256, classification_classes*objects_classes, name='yolo_output_1', softmax_activation=True)(x)         

            x = self.yolo_conv(64, name='yolo_conv_2')((x, x_36))
            output_2 = self.yolo_output(128, bbox_classes*objects_classes, name='yolo_output_2')(x)

            return Model(inputs, (output_0, output_1, output_2), name='yolov3')


    # def build_feature_extractor(self, inputs):

    #     x = tf.keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=self.input_size)(inputs)
    #     x = tf.keras.layers.AveragePooling2D(2,2)(x)

    #     x = tf.keras.layers.Conv2D(32, kernel_size=3, activation = 'relu')(x)
    #     x = tf.keras.layers.AveragePooling2D(2,2)(x)

    #     x = tf.keras.layers.Conv2D(64, kernel_size=3, activation = 'relu')(x)
    #     x = tf.keras.layers.AveragePooling2D(2,2)(x)

    #     return x

    # def build_model_adaptor(self, inputs):
    #     x = tf.keras.layers.Flatten()(inputs)
    #     x = tf.keras.layers.Dense(64, activation='relu')(x)
    #     return x

    # def build_classifier_head(self, inputs):
    #     x = tf.keras.layers.Dense(self.h.num_classes*self.h.grid_partition*self.h.grid_partition)(inputs)
    #     x = tf.keras.layers.Reshape((self.h.grid_partition, self.h.grid_partition, self.h.num_classes))(x)
    #     return tf.keras.layers.Softmax(name = 'classifier_head')(x)

    # def build_regressor_head(self, inputs):
    #     x = tf.keras.layers.Dense(self.h.bbox_number*self.h.grid_partition*self.h.grid_partition, activation='relu')(inputs)
    #     return tf.keras.layers.Reshape((self.h.grid_partition, self.h.grid_partition, self.h.bbox_number), name = 'regressor_head')(x)

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
        print('y train - ', y_train)

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
            test_loss, test_class_loss, test_detect_loss, test_class_acc, test_detect_mse  = self.model.evaluate(X_test, y_test)
            print('test loss -- ', test_loss)
            print('Tested classification loss:', test_class_loss)
            print('Tested detect loss:', test_detect_loss)
            print('Tested classification accuracy:', test_class_acc)
            print('Tested detect mse:', test_detect_mse)

        # Save the model.
        self.model.save(os.path.join(self.h.save_model_folder, self.h.save_model_name))
    

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


        # Resize the image.
        resized_image = utils.format_image(img, self.h.image_size)

        # Normalize the image.
        binary_image = resized_image.astype('float32') / 255

        # Reshape the image.
        reshape_image = binary_image.reshape((1, self.h.image_size, self.h.image_size, 1))

        # Make the prediction.
        result = self.model.predict(reshape_image)

        return result


if __name__ == '__main__':
    h_conf = parameters.get_config()
    boxer = bbox_ai(h_conf)
