import os
import sys
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import csv
from pathlib import Path
from datetime import datetime
import logging

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

def main():

    path_data = os.path.join(os.path.curdir, "data/FactinMorphology_analysis")
    path_output = os.path.join(os.path.curdir, f"analysis-{current_time}")
    path_model = os.path.join(os.path.curdir, "trained_models")

    model = load_model(os.path.join(path_model, 'platelet_classifier.h5'))


    class_labels = {0:"Hollow", 1:"Nodules", 2:"Solid"}

    SIZE_ROWS = 299 # 299 for inceptionv3
    SIZE_COLS = 299 # 96 for custom model
    SIZE_CHAN = 3 # use 1 for custom model, 3 for pretrained model

    tol = 1 # % of pixels to saturate (i.e. 5 means keep pixels within 5-95%)

    image_names = [x.name for x in list(os.scandir(path_data)) if x.is_file() and os.path.splitext(x)[1] == ".png"]
    image_names.sort()
    num_images = len(image_names)
    logger.info("Found %d images to analyze.", num_images)
    X_data = np.ndarray((num_images, SIZE_ROWS, SIZE_COLS, SIZE_CHAN), dtype=np.uint8)

    for i, image_name in enumerate(image_names):
        path_image_file = os.path.join(path_data, image_name)
        img = cv2.imread(path_image_file, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (40, 40))

        # image corrections (contrast 5-95 percentile)
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        cumhist = np.cumsum(hist)
        total = img.size
        low_bound = total * tol / 100  # low number of pixels to remove
        upp_bound = total * (100-tol) / 100 # upp number of pixels to remove
        lowb = None
        uppb = None
        for k, h in enumerate(cumhist):
            if h > low_bound and not lowb:
                lowb = k
            if h > upp_bound and not uppb:
                uppb = k-1
        img = np.clip(img, lowb, uppb)
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

        img = cv2.resize(img, (SIZE_ROWS, SIZE_COLS))
        if SIZE_CHAN == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif SIZE_CHAN == 1:
            img = img.reshape(SIZE_ROWS, SIZE_COLS, 1)

        X_data[i] = np.array(img)

    logger.info("Processed %d images.", len(X_data))

    X_datagen = ImageDataGenerator(rescale=1./255)
    X_datagen.fit(X_data, augment=True)
    X_generator = X_datagen.flow(
        X_data,
        batch_size=1,
        shuffle=False)


    memory_info = tf.config.experimental.get_memory_info('GPU:0')
    print(memory_info)

    predictions = model.predict(X_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    counts_classes = np.bincount(predicted_classes)

    if ~os.path.exists(path_output):
        os.mkdir(path_output)

    path_output_file = os.path.join(path_output, 'classifications.csv')

    logger.info("Successfully completed image classifications.\n" + \
        "\tWriting output to %s", path_output_file)

    with open(path_output_file, 'w', newline='') as f1:
        writer1 = csv.writer(f1)
        writer1.writerow(['Total', class_labels[0], class_labels[1], class_labels[2]])
        writer1.writerow(['', counts_classes[0], counts_classes[1], counts_classes[2]])
        writer1.writerow(['filename', 'classification', 'max class', class_labels[0], class_labels[1], class_labels[2]])
        for i in range(len(predictions)):
            cv2.imwrite(os.path.join(path_output, 'p' + class_labels[predicted_classes[i]] + f'{predictions[i][predicted_classes[i]]:.2f}' + '_' + image_names[i]), X_data[i,:,:,0])
            writer1.writerow([image_names[i], class_labels[predicted_classes[i]], predictions[i][predicted_classes[i]]] + list(predictions[i]))

    # for i in range(len(predictions)):
    #     img = X_data[i]
    #     cv2.imshow('image', img)
    #     cv2.setWindowTitle('image', f'{class_labels[predicted_classes[i]]} {predictions[i][predicted_classes[i]]:.3f}')
    #     cv2.waitKey(0)

if __name__ == "__main__":
    main()
