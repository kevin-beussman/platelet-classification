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
tf.keras.utils.set_random_seed(1313)

def main():
    global min_loss

    path_data = "C:/Users/kevin/git-workspace/tf-platelets/Data/"
    path_output = "C:/Users/kevin/git-workspace/tf-platelets/output/"
    # path_data = "/mmfs1/home/beussk/platelet_factin_tf/Data/"
    # path_output = "/mmfs1/home/beussk/platelet_factin_tf/output/"

    SIZE_ROWS = 299 # 299 for inceptionv3
    SIZE_COLS = 299 # 96 for custom model
    SIZE_CHAN = 3 # use 1 for custom model, 3 for pretrained model

    epochs = 500  # total epochs to run up to
    batch_size = 32  # number of images per batch
    subset = float('inf') # float('inf')

    initial_learning_rate = 0.001
    final_learning_rate = 0.00000001

    load_checkpoint = True

    custom_model = False
    fine_tune = True

    #####################
    batch_size = batch_size
    decay_rate = tfmath.log(final_learning_rate /
                            initial_learning_rate)/(epochs-1)

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
        # def right_angle_rotate(input_image):
        #     angle = np.random.choice([0, 90, 180, 270])
        #     if angle != 0:
        #         input_image = apply_affine_transform(
        #             input_image, theta=angle, fill_mode='nearest')
        #     return input_image

        # train_datagen = ImageDataGenerator(
        #     fill_mode = 'nearest',
        #     preprocessing_function = right_angle_rotate,
        #     rescale = 1./255, # i.e. get all pixels betwen 0-1, works for 8-bit image uint8 = 255
        #     horizontal_flip = True, # mirror left/right
        #     vertical_flip = True # mirror up/down
        #     )

        # train_datagen = ImageDataGenerator(
        #     rescale=1./255, # i.e. get all pixels betwen 0-1, works for 8-bit image uint8 = 255
        #     rotation_range=45, # degrees to rotate image (from -45 to +45 degrees)
        #     width_shift_range=0.1, # shift left/right by up to 10% image width
        #     height_shift_range=0.1, # shift up/down by up to 10% image width
        #     shear_range=0.1, # applies some image shearing transformations
        #     zoom_range=0.5, # applies random zoom (e.g. if set to 0.5, yields zoom 0.5-1.5x)
        #     horizontal_flip=True, # mirror left/right
        #     vertical_flip=True, # mirror up/down
        #     # brightness_range=[0.8,1.0], # modifies brightness
        #     fill_mode='nearest')

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

    def define_custom_model():
        model = Sequential()
        model.add(BatchNormalization(input_shape=(SIZE_ROWS, SIZE_COLS, SIZE_CHAN)))
        model.add(Convolution2D(64, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Convolution2D(128, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(BatchNormalization())
        model.add(Convolution2D(256, (5, 5), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(GlobalAveragePooling2D())
        model.add(Dense(256, activation='relu'))
        # model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(class_labels), activation='softmax'))
        # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # lr_schedule = ExponentialDecay(learning_rate, decay_steps, decay_rate)

        # learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
        # steps_per_epoch = int(train_generator.n/batch_size)
        # lr_schedule = ExponentialDecay(
        #                 initial_learning_rate=initial_learning_rate,
        #                 decay_steps=steps_per_epoch,
        #                 decay_rate=learning_rate_decay_factor,
        #                 staircase=True)

        # model.compile(optimizer=SGD(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])

        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def define_pretrained_model():
        # InceptionV3
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(SIZE_ROWS, SIZE_COLS, SIZE_CHAN))
        # base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv5_block3_3_bn').output)

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.5)(x)

        predictions = Dense(len(class_labels), activation='softmax', name='predictions')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # learning_rate_decay_factor = (final_learning_rate / initial_learning_rate)**(1/epochs)
        # steps_per_epoch = int(train_generator.n/batch_size)
        # lr_schedule = ExponentialDecay(
        #                 initial_learning_rate=initial_learning_rate,
        #                 decay_steps=steps_per_epoch,
        #                 decay_rate=learning_rate_decay_factor,
        #                 staircase=True)

        # model.compile(optimizer=SGD(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])

        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    X_train, Y_train, X_val, Y_val, class_labels, class_weights = load_data(path_data)

    train_generator, val_generator = augment_data(X_train, Y_train, X_val, Y_val)

    if load_checkpoint and os.path.exists(os.path.join(path_output, 'checkpoint.h5')):
        print("Loading from existing checkpoint...")
        model = load_model(os.path.join(path_output, 'checkpoint.h5'))

        last_save = np.load(os.path.join(path_output, 'last_save.npy'), allow_pickle=True)
        last_save = last_save.item()

        initial_epoch = last_save['epoch']
        # initial_learning_rate = last_save['lr']
        min_loss = last_save['val_loss']

        csv_callback = CSVLogger(os.path.join(path_output, 'history.csv'),
                                 separator=',',
                                 append=True)
    else:
        if not os.path.exists(path_output):
            os.mkdir(path_output)
        if custom_model:
            model = define_custom_model()
        else:
            model = define_pretrained_model()
        initial_epoch = 0
        min_loss = float('inf')

        model.summary()

        predictions = model.predict(val_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        actuals = val_generator.y
        actual_classes = np.argmax(actuals, axis=1)
        confusion = tfmath.confusion_matrix(actual_classes, predicted_classes)
        print('Initial untrained validation predictions:')
        print(class_labels)
        print(confusion.numpy()) # predictions on top, actual on right

        csv_callback = CSVLogger(os.path.join(path_output, 'history.csv'),
                                 separator=',',
                                 append=False)
        # plot_model(model, to_file='test.png', show_shapes=True, show_layer_names=False)

    # if not os.path.exists(path_output):
    #     os.mkdir(path_output)
    # model = define_model()
    # initial_epoch = 0

    # model.summary()

    # cp_callback = ModelCheckpoint(
    #     # filepath = os.path.join(path_output, 'checkpoint_weights_{epoch:02d}-{val_loss:.2f}.h5'),
    #     filepath = os.path.join(path_output, 'checkpoint.h5'),
    #     save_best_only = True,
    #     monitor = 'val_loss',
    #     mode = 'min',
    #     verbose = 1)

    # reduce_lr = ReduceLROnPlateau(
    #     monitor = 'val_loss',
    #     factor = 0.5,
    #     patience = 3,
    #     min_lr = final_learning_rate,
    #     verbose = 2)

    # def lr_exp_decay(epoch, lr):
    #     return initial_learning_rate * tfmath.exp(decay_rate*epoch)

    # reduce_lr = LearningRateScheduler(lr_exp_decay, verbose=1)

    class CustomCallback(Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch < epochs:
                new_lr = initial_learning_rate * tfmath.exp(decay_rate*epoch)
            else:
                new_lr = final_learning_rate # this is used for fine-tuning
            tf.keras.backend.set_value(model.optimizer.lr, new_lr)

        def on_epoch_end(self, epoch, logs=None):
            global min_loss

            current_lr = model.optimizer.lr.numpy()
            print(f"epoch = {epoch + 1}, learning rate = {current_lr}")

            if logs['val_loss'] < min_loss:
                print(f"loss = {logs['val_loss']}, saving model")
                model.save(os.path.join(path_output, 'checkpoint.h5'))
                outputs = {'epoch': epoch + 1, 'lr': current_lr,
                           'val_loss': logs['val_loss']}
                np.save(os.path.join(path_output, 'last_save.npy'), outputs)
                min_loss = logs['val_loss']

    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.n // batch_size),
        epochs=epochs,
        initial_epoch=initial_epoch,
        validation_data=val_generator,
        validation_steps=max(1, val_generator.n // batch_size),
        callbacks=[csv_callback, CustomCallback()],
        class_weight=class_weights,
        verbose=1
    )

    if not custom_model and fine_tune:
        for layer in model.layers[279:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
            # if layer.__module__ is not 'keras.layers.normalization.batch_normalization':
            # might need to skip the batch normalization layers
                layer.trainable = True
            
        model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
        
        csv_callback.append = True

        history_fine = model.fit(
            train_generator,
            steps_per_epoch=max(1, train_generator.n // batch_size),
            epochs=2*epochs,
            initial_epoch=initial_epoch,
            validation_data=val_generator,
            validation_steps=max(1, val_generator.n // batch_size),
            callbacks=[csv_callback, CustomCallback()],
            class_weight=class_weights,
            verbose=1
        )

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
