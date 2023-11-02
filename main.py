# Import necessary libraries
import parameters
import model
import utils
import cv2 as cv
import numpy as np

def main():

    # If training, then train the model.
    h_parameters = parameters.get_config()
    bbox_ai = model.bbox_ai(h_parameters)

    print('start training')
    bbox_ai.train()

    print('predicting')
    img = cv.imread('43.jpg', cv.IMREAD_GRAYSCALE)
    formated_img = utils.format_image(img, h_parameters.image_size)
    result = bbox_ai.predict(img)
    objects = result[0]
    classification = result[1]
    detecting = result[2]
    for x, row in enumerate(objects[0]):
        for y, classes in enumerate(row):
            index_obj = np.argmax(classes)
            if index_obj != 0 and classes[index_obj] > 0.5:
                index_classification = np.argmax(classification[0][x][y])
                class_name = utils.get_key_from_dict(h_parameters.image_classes[0], index_classification)
                grid_size = h_parameters.image_size/h_parameters.grid_partition
                formated_img_txt = cv.putText(formated_img, class_name, ((int(x*grid_size), int(y*grid_size))), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

                bbox = detecting[0][x][y]
                bbox = bbox*h_parameters.image_size
                x_px, y_px, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                x_start = int(x_px-w)
                if x_start < 0:
                    x_start = 0
                elif x_start > h_parameters.image_size:
                    x_start = h_parameters.image_size
                y_start = int(y_px-h)
                if y_start < 0:
                    y_start = 0
                elif y_start > h_parameters.image_size:
                    y_start = h_parameters.image_size
                x_end = int(x_px+w)
                if x_end < 0:
                    x_end = 0
                elif x_end > h_parameters.image_size:
                    x_end = h_parameters.image_size
                y_end = int(y_px+h)
                if y_end < 0:
                    y_end = 0
                elif y_end > h_parameters.image_size:
                    y_end = h_parameters.image_size
                formated_img_rect = cv.rectangle(formated_img_txt, (x_start, y_start), (x_end, y_end), (0,255,0), 1)
            else:
                formated_img_rect = formated_img

    cv.imshow('classes', formated_img_rect)
    cv.waitKey(0)

if __name__=="__main__":
    main()
