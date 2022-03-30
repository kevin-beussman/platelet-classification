import os
import cv2
import tensorflow as tf
import numpy as np
# import pandas as pd
from tensorflow.keras.utils import to_categorical  # , plot_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.callbacks import CSVLogger, Callback
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator, apply_affine_transform
from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import InceptionV3
from tensorflow import math as tfmath
# from sklearn.model_selection import train_test_split
import csv

np.random.seed(1313)
tf.random.set_seed(1313)

def main():
    global min_loss

    path_data = "C:/Users/kevin/git-workspace/tf-platelets/Data/"
    # path_output = "C:/Users/kevin/git-workspace/tf-platelets/outputtest/"
    path_output = "C:/Users/kevin/OneDrive/Desktop/test3/"
    # path_data = "/mmfs1/home/beussk/platelet_factin_tf/Data/"
    # path_output = "/mmfs1/home/beussk/platelet_factin_tf/output3/"

    batch_size = 32  # number of images per batch

    SIZE_ROWS = 299 # 299 for inceptionv3
    SIZE_COLS = 299 # 96 for custom model
    SIZE_CHAN = 3 # use 1 for custom model, 3 for pretrained model

    tol = 1 # % of pixels to saturate (i.e. 5 means keep pixels within 5-95%)

    def load_data(import_dir):
        val_dir = os.path.join(import_dir,'Validation')
        data_labels = os.listdir(val_dir)
        data_labels.sort()

        num_val_samples = []
        for label in data_labels:
            num_val_samples.append(len([1 for x in list(os.scandir(os.path.join(val_dir, label))) if x.is_file() and os.path.splitext(x)[1] == ".png"]))
        tot_val_samples = sum(num_val_samples)

        X_val_data = np.ndarray((tot_val_samples, SIZE_ROWS, SIZE_COLS, SIZE_CHAN), dtype=np.uint8)
        Y_val_data = np.ndarray((tot_val_samples,), dtype=np.uint8)

        filenames = []
        for j, label in enumerate(data_labels):
            val_image_names = [x for x in list(os.scandir(os.path.join(val_dir, label))) if x.is_file() and os.path.splitext(x)[1] == ".png"]
            filenames += val_image_names
            print(f'Loading {label}...', end='')
            for i, image_name in enumerate(val_image_names):
                
                # img = cv2.imread(os.path.join(import_dir, label, image_name), cv2.IMREAD_COLOR)
                img = cv2.imread(os.path.join(import_dir, label, image_name), cv2.IMREAD_GRAYSCALE)

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

                idx = i + sum(num_val_samples[:j])
                X_val_data[idx] = np.array(img)
                Y_val_data[idx] = j
            print(f'Done ({num_val_samples[j]} validation images)')
        Y_val_data = to_categorical(Y_val_data, len(num_val_samples))

        return X_val_data, Y_val_data, data_labels, filenames

    X_val, Y_val, class_labels, filenames = load_data(path_data)

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen.fit(X_val, augment=True)
    val_generator = val_datagen.flow(
            X_val, Y_val,
            batch_size=batch_size,
            shuffle=False)

    model = load_model(os.path.join(path_output, 'platelet_classifier.h5'))

    predictions = model.predict(val_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    actuals = val_generator.y
    actual_classes = np.argmax(actuals, axis=1)

    if not os.path.exists(os.path.join(path_output, 'misclassified_images')):
        os.mkdir(os.path.join(path_output, 'misclassified_images'))
    with open(os.path.join(path_output, 'misclassified.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prediction ' + class_labels[0], class_labels[1], class_labels[2], 'actual ' + class_labels[0], class_labels[1], class_labels[2]])
        misclassified_n = 0
        for i in range(len(predictions)):
            if predicted_classes[i] != actual_classes[i]:
                misclassified_n += 1
                # print([filenames[i].path] + list(predictions[i]) + list(actuals[i]))
                # cv2.imshow(X_val[i])
                cv2.imwrite(os.path.join(path_output, 'misclassified_images', 'p' + class_labels[predicted_classes[i]] + f'{predictions[i][predicted_classes[i]]:.2f}' + '_a' + class_labels[actual_classes[i]] + '_' + filenames[i].name), X_val[i,:,:,0])
                writer.writerow([filenames[i].path] + list(predictions[i]) + list(actuals[i]))

    confusion = tfmath.confusion_matrix(actual_classes, predicted_classes)
    # np.save(os.path.join(path_output, 'val_metrics.npy'), 
    #         {'labels': class_labels, 'confusion_matrix': confusion.numpy()})
    print('Final trained validation predictions:')
    print(class_labels)
    print(confusion.numpy())

    print(f'accuracy = {(len(predictions) - misclassified_n) / len(predictions)}')

if __name__ == "__main__":
    main()
