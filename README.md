    Military aircraft object detection project

    Main goal of this project is to detecting and classify aircrafts on satelites images
to analyzing in future.

For this purpose we will use Military Aircraft Recognition dataset.

The Military Aircraft Recognition dataset is a remote sensing image dataset that includes 3,842 images, 20 types, and 22,341 instances annotated with horizontal bounding boxes and oriented bounding boxes. The images are of various military aircraft, including fighter jets, bombers, transport aircraft, and helicopters. The dataset is split into training, validation, and test sets.

The dataset can be used for a variety of tasks, including:

Military aircraft recognition: The dataset can be used to train machine learning models to recognize different types of military aircraft. This can be useful for military applications, such as target identification and tracking.
Aircraft detection: The dataset can also be used to train machine learning models to detect aircraft in images and videos. This can be useful for a variety of applications, such as traffic monitoring and surveillance.
Aircraft tracking: The dataset can also be used to train machine learning models to track aircraft over time. This can be useful for a variety of applications, such as flight control and air traffic management.
The dataset is licensed under the Apache 2.0 license, which means that it is free to use and distribute.

Firstly explore dataset in jupyter notebook. The file with jupiter notebook on git and has name: "data_exploratory.ipynb"

In jupyter we check what are in dataset folder. There are three folders with anotation to each images - Annotations, with
file names divided in to two files test.txt and train.txt. test.txt has 2511 files, in train.txt there 1331 files.
some notes has in filename space so when you use it you need filter or something like that.
Next we see what inside files with annotations. Some file names in horizontal bounding box annotations has extension not ".jpg" but ".xml" in future program its could cause issue so you need also use some filters or "if".
Next we see some statistic to understand what information is useless. This type of information that don`t changing over dataset we could check as useless.

After we analyze dataset we start looking for model architecture. The best model architecrute for now its YOLO so we can take this architecture as backbone but simpler because we have not enough time.

Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to
==================================================================================================
 input_1 (InputLayer)        [(None, 460, 460, 1)]        0         []

 conv2d (Conv2D)             (None, 458, 458, 16)         160       ['input_1[0][0]']

 average_pooling2d (Average  (None, 229, 229, 16)         0         ['conv2d[0][0]']
 Pooling2D)

 conv2d_1 (Conv2D)           (None, 227, 227, 32)         4640      ['average_pooling2d[0][0]']

 average_pooling2d_1 (Avera  (None, 113, 113, 32)         0         ['conv2d_1[0][0]']
 gePooling2D)

 conv2d_2 (Conv2D)           (None, 111, 111, 64)         18496     ['average_pooling2d_1[0][0]']

 average_pooling2d_2 (Avera  (None, 55, 55, 64)           0         ['conv2d_2[0][0]']
 gePooling2D)

 flatten (Flatten)           (None, 193600)               0         ['average_pooling2d_2[0][0]']

 dense (Dense)               (None, 64)                   1239046   ['flatten[0][0]']
                                                          4

 classifier_head (Dense)     (None, 20)                   1300      ['dense[0][0]']

 regressor_head (Dense)      (None, 4)                    260       ['dense[0][0]']

==================================================================================================
Total params: 12415320 (47.36 MB)
Trainable params: 12415320 (47.36 MB)
Non-trainable params: 0 (0.00 Byte)


In the model we have two output layer because one for classification and other for horizontal object detection.


The structure of our project include model.py for our model, preprocessing.py to prepare our dataset for training, parameters.py for all parameters of our project, data_loader.py to loading data and main.py to start training.
