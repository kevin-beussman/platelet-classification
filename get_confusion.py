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

np.random.seed(1313)
# tf.keras.utils.set_random_seed(1313)
tf.random.set_seed(1313)

def main():
    global min_loss

    # path_data = "C:/Users/kevin/git-workspace/tf-platelets/Data/"
    # path_output = "C:/Users/kevin/git-workspace/tf-platelets/outputtest/"
    path_data = "/mmfs1/home/beussk/platelet_factin_tf/Data/"
    path_output = "/mmfs1/home/beussk/platelet_factin_tf/output3/"

    batch_size = 32  # number of images per batch

    custom_model = False

    if custom_model:
        SIZE_ROWS = 96 # 299 for inceptionv3
        SIZE_COLS = 96 # 96 for custom model
        SIZE_CHAN = 1 # use 1 for custom model, 3 for pretrained model
    else:
        SIZE_ROWS = 299 # 299 for inceptionv3
        SIZE_COLS = 299 # 96 for custom model
        SIZE_CHAN = 3 # use 1 for custom model, 3 for pretrained model

    def load_data(import_dir):
        train_dir = os.path.join(import_dir,'Training')
        val_dir = os.path.join(import_dir,'Validation')
        data_labels = os.listdir(train_dir)

        num_train_samples = []
        num_val_samples = []
        for label in data_labels:
            num_train_samples.append(len([1 for x in list(os.scandir(os.path.join(train_dir, label))) if x.is_file() and os.path.splitext(x)[1] == ".png"]))
            num_val_samples.append(len([1 for x in list(os.scandir(os.path.join(val_dir, label))) if x.is_file() and os.path.splitext(x)[1] == ".png"]))
        tot_train_samples = sum(num_train_samples)
        tot_val_samples = sum(num_val_samples)

        X_train_data = np.ndarray((tot_train_samples, SIZE_ROWS, SIZE_COLS, SIZE_CHAN), dtype=np.uint8)
        Y_train_data = np.ndarray((tot_train_samples,), dtype=np.uint8)
        X_val_data = np.ndarray((tot_val_samples, SIZE_ROWS, SIZE_COLS, SIZE_CHAN), dtype=np.uint8)
        Y_val_data = np.ndarray((tot_val_samples,), dtype=np.uint8)

        for j, label in enumerate(data_labels):
            pct = 0
            train_image_names = [x for x in list(os.scandir(os.path.join(train_dir, label))) if x.is_file() and os.path.splitext(x)[1] == ".png"]
            val_image_names = [x for x in list(os.scandir(os.path.join(val_dir, label))) if x.is_file() and os.path.splitext(x)[1] == ".png"]
            
            print(f'Loading {label}...', end='')
            for i, image_name in enumerate(train_image_names):
                
                # img = cv2.imread(os.path.join(import_dir, label, image_name), cv2.IMREAD_COLOR)
                img = cv2.imread(os.path.join(import_dir, label, image_name), cv2.IMREAD_GRAYSCALE)

                # image corrections (contrast 5-95 percentile)
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                cumhist = np.cumsum(hist)
                tol = 5  # % of pixels to saturate (i.e. keep 5-95%)
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

                idx = i + sum(num_train_samples[:j])
                X_train_data[idx] = np.array(img)
                Y_train_data[idx] = j
            
            for i, image_name in enumerate(val_image_names):
                
                # img = cv2.imread(os.path.join(import_dir, label, image_name), cv2.IMREAD_COLOR)
                img = cv2.imread(os.path.join(import_dir, label, image_name), cv2.IMREAD_GRAYSCALE)

                # image corrections (contrast 5-95 percentile)
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                cumhist = np.cumsum(hist)
                tol = 5  # % of pixels to saturate (i.e. keep 5-95%)
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
            print(f'Done ({num_train_samples[j]} training images, {num_val_samples[j]} validation images)')
        Y_train_data = to_categorical(Y_train_data, len(num_train_samples))
        Y_val_data = to_categorical(Y_val_data, len(num_val_samples))

        data_weights = {}
        for j, label in enumerate(data_labels):
            data_weights[j] = (1 / num_train_samples[j])*(tot_train_samples / 2.0)

        return X_train_data, Y_train_data, X_val_data, Y_val_data, data_labels, data_weights

    def augment_data(X_train, Y_train, X_val, Y_val):
        train_datagen = ImageDataGenerator(
            fill_mode='nearest',
            rescale=1./255,  # i.e. get all pixels betwen 0-1, works for 8-bit image uint8 = 255
            # degrees to rotate image (from -45 to +45 degrees)
            rotation_range=45,
            horizontal_flip=True,  # mirror left/right
            vertical_flip=True  # mirror up/down
        )

        # don't modify validation images
        val_datagen = ImageDataGenerator(rescale=1./255)

        train_datagen.fit(X_train, augment=True)

        val_datagen.fit(X_val, augment=True)

        train_generator = train_datagen.flow(
            X_train, Y_train,
            batch_size=batch_size)

        val_generator = val_datagen.flow(
            X_val, Y_val,
            batch_size=batch_size,
            shuffle=False)

        return train_generator, val_generator

    X_train, Y_train, X_val, Y_val, class_labels, class_weights = load_data(path_data)

    train_generator, val_generator = augment_data(X_train, Y_train, X_val, Y_val)

    model = load_model(os.path.join(path_output, 'checkpoint.h5'))

    predictions = model.predict(val_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    actuals = val_generator.y
    actual_classes = np.argmax(actuals, axis=1)
    confusion = tfmath.confusion_matrix(actual_classes, predicted_classes)
    np.save(os.path.join(path_output, 'val_metrics.npy'), 
            {'labels': class_labels, 'confusion_matrix': confusion.numpy()})
    print('Final trained validation predictions:')
    print(class_labels)
    print(confusion.numpy())

if __name__ == "__main__":
    main()
