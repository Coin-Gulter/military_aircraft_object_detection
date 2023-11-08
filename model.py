import numpy as np
import cv2 as cv
import os
import data_loader
import utils
import parameters
from itertools import repeat
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, UpSampling2D, ZeroPadding2D, LeakyReLU, Add, Concatenate, Input, Softmax, Activation, Reshape, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Recall, Precision, Accuracy, F1Score

 
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """A custom BatchNormalization layer that can be used in inference mode without updating the moving mean and moving variance."""
    def call(self, x, training=False):
        """Computes the output of the layer.

        Args:
            x: The input tensor.
            training: A Boolean value indicating whether the layer is in training mode.

        Returns:
            The output tensor.
        """
        if training is None: training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

class bbox_ai():

    def __init__(self, h_parameters):

        self.h = h_parameters
        self.input_size = (self.h.image_size, self.h.image_size, 1)
        self.metric_acc = tf.keras.metrics.Accuracy()
        self.metric_precision = Precision()
        self.metric_recall = Recall()
        self.metric_objects_f1 = F1Score()
        self.metric_class_f1 = F1Score()

        # Create the model.
        print('Creating model...')
        self.model = self.yolo_v3(self.h.image_size, self.h.objects_number, self.h.num_classes, self.h.bbox_number)
        print('Model created')

        # Check if the pretrained model is used.
        if self.h.pretrained:
            print('Loading pretrained model...')
            self.model.load_weights(os.path.join(self.h.save_model_folder, self.h.save_model_name))
            print('Model loaded')

        # Summarize the model.
        self.model.summary()

    def backbone_conv(self, x, filters, size, strides=1, batch_norm=True):
        # Helper function for convolutional layers in the backbone.
        if strides == 1:
            padding = 'same'
        else:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)
            padding = 'valid'
        x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding, 
                   use_bias=not batch_norm, kernel_regularizer=l2(0.001))(x)
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x
    
    def backbone_res(self, x, filters):
        # Helper function for residual blocks in the backbone.
        previous = x
        x = self.backbone_conv(x, filters, 2, 1)
        x = self.backbone_conv(x, filters, 3)
        x = Add()([previous , x])
        return x
    
    def backbone_block(self, x, filters, blocks):
        # Helper function for creating a block of layers in the backbone.
        x = self.backbone_conv(x, filters, 3, strides=2)
        for _ in repeat(None, blocks):
            x = self.backbone_res(x, filters)
        return x
    
    def darknet(self, name=None):
        # Definition of the Darknet architecture, a backbone network.
        x = inputs = Input([self.h.image_size, self.h.image_size, self.h.input_channels])
        x = self.backbone_conv(x, 16, 3)
        x = self.backbone_block(x, 32, 1)
        x = self.backbone_block(x, 64, 2)
        x = x_36 = self.backbone_block(x, 128, 4)
        x = x_61 = self.backbone_block(x, 256, 4)
        x = self.backbone_block(x, 512, 2)
        return tf.keras.Model(inputs, (x_36, x_61, x), name=name)
    
    def yolo_conv(self, filters, name=None):
        # YOLO convolutional layer with skip connections.
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
            x = Dropout(rate=self.h.lr_dropout)(x)
            x = self.backbone_conv(x, filters, 1)
            x = self.backbone_conv(x, filters * 2, 3)
            x = self.backbone_conv(x, filters, 1)
            x = Dropout(rate=self.h.lr_dropout)(x)
            return Model(inputs, x, name=name)(x_in)
        return yolo_conv

    def yolo_output(self, filters, classes, name=None, max_pool_number=0, softmax_activation=False, sigmoid_activation=False):
        # YOLO output layer for object detection.
        def yolo_output(x_in):
            x = inputs = Input(x_in.shape[1:])
            x = self.backbone_conv(x, filters * 2, 3)
            x = self.backbone_conv(x, classes, 1, batch_norm=False)

            if max_pool_number > 0:
                x = MaxPooling2D(pool_size=(2*max_pool_number, 2*max_pool_number))(x)

            if softmax_activation:
                x = Softmax()(x)
            elif sigmoid_activation:
                x = Activation(activation='sigmoid')(x)

            return tf.keras.Model(inputs, x, name=name)(x_in)
        return yolo_output

    def yolo_v3(self, size, objects_classes, classification_classes, bbox_classes):
        # YOLOv3 model architecture combining the backbone and output layers.
            x = inputs = Input([size, size, self.h.input_channels])

            x_36, x_61, x = self.darknet(name='yolo_darknet')(x)

            x = self.yolo_conv(256, name='yolo_conv_0')(x)
            obj_exist_out = self.yolo_output(512, objects_classes+1, name='obj_exist', sigmoid_activation=True)(x)

            x = self.yolo_conv(128, name='yolo_conv_1')((x, x_61))
            obj_class_out = self.yolo_output(256, classification_classes*objects_classes, name='obj_classification', max_pool_number=1, softmax_activation=True)(x)         

            x = self.yolo_conv(64, name='yolo_conv_2')((x, x_36))
            obj_detect_out = self.yolo_output(128, bbox_classes*objects_classes, max_pool_number=2, name='obj_detection')(x)

            return Model(inputs, (obj_exist_out, obj_class_out, obj_detect_out), name='yolov3')

    def my_accuracy(self, y_true, y_pred):
        self.metric_acc.update_state(tf.argmax(y_true), tf.argmax(y_pred))
        acc = self.metric_acc.result()
        self.metric_acc.reset_state()
        return acc

    def my_precision(self, y_true, y_pred):
        self.metric_precision.update_state(y_true, y_pred)
        acc = self.metric_precision.result()
        self.metric_precision.reset_state()
        return acc
    
    def my_recall(self, y_true, y_pred):
        self.metric_recall.update_state(y_true, y_pred)
        acc = self.metric_recall.result()
        self.metric_recall.reset_state()
        return acc

    def my_f1_score(self, y_true, y_pred):
        f1 = 2*(self.my_recall(y_true, y_pred) * self.my_precision(y_true, y_pred)) / \
            (self.my_recall(y_true, y_pred) + self.my_precision(y_true, y_pred))
        return f1



    def train(self):
        # Method to train the model.

        # Get model output shapes
        obj_exist_shape = self.model.get_layer('obj_exist').output
        self.h.grid_partition = obj_exist_shape.shape[1]

        # Load the training and test data.
        X_train, y_train, X_test, y_test = data_loader.loading_data(self.h)
        # print('y train - ', y_train)

        # Compile the model.
        self.model.compile(optimizer=self.h.optimizer, 
                           loss={'obj_exist' : 'binary_crossentropy', 'obj_classification' : 'categorical_crossentropy', 'obj_detection' : 'mse' }, 
                           metrics={'obj_exist' : [ self.my_accuracy, self.my_precision, self.my_recall, self.my_f1_score], 
                                    'obj_classification' : [ self.my_accuracy, self.my_precision, self.my_recall, self.my_f1_score], 
                                    'obj_detection' : 'mse' })

        # Train the model.
        self.model.fit(X_train, 
                        y_train, 
                        epochs=self.h.num_epochs, 
                        batch_size=self.h.batch_size, 
                        validation_split=self.h.validation_split)

        # Evaluate the model on the test set.
        if self.h.with_test:
            test_loss, test_obj_loss, test_class_loss, test_detect_loss, test_obj_acc, _, _, test_obj_f1_score, test_class_acc, _, _, test_class_f1_score, test_detect_mse = self.model.evaluate(X_test, y_test)
            print('test loss -- ', test_loss)
            print('Tested objects loss:', test_obj_loss)
            print('Tested classification loss:', test_class_loss)
            print('Tested detect loss:', test_detect_loss)
            print('Tested objects accuracy:', test_obj_acc)
            print('Tested objects f1 score:', test_obj_f1_score)
            print('Tested classification accuracy:', test_class_acc)
            print('Tested objects f1 score:', test_class_f1_score)
            print('Tested detect mse:', test_detect_mse)

        # Save the model.
        self.model.save_weights(os.path.join(self.h.save_model_folder, self.h.save_model_name))
    

    def predict(self, img:np.ndarray):
        # Method to make predictions on input images.

        formated_img = utils.format_image(img, self.h.image_size, self.h.input_channels)

        # Normalize the image.
        binary_image = formated_img.astype('float32') / 255

        # Reshape the image.
        reshape_image = binary_image.reshape((1, self.h.image_size, self.h.image_size, self.h.input_channels))

        # Make the prediction.
        result = self.model.predict(reshape_image)

        return result


if __name__ == '__main__':
    h_conf = parameters.get_config()
    boxer = bbox_ai(h_conf)
