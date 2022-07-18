import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import csv
from pathlib import Path

def main():

    path_data = "G:/.shortcut-targets-by-id/1SYh5QCYA01sjD05rNMxSCwqUTrxombOD/Platelet higher resolution analysis - collab with Adithan Kandasamy, Kevin Beussman, Molly Mollica/New data - May 2022/FactinMorphology_analysis"
    # path_output = path_data
    path_output = "C:/Users/kevin/OneDrive/Desktop/test2/"

    # path_model = "C:/Users/kevin/git-workspace/tf-platelets/"
    path_model = "C:/Users/kevin/OneDrive/Desktop/output2022-06-08_2"

    model = load_model(os.path.join(path_model, 'platelet_classifier.h5'))
    

    class_labels = {0:"Hollow", 1:"Nodules", 2:"Solid"}

    SIZE_ROWS = 299 # 299 for inceptionv3
    SIZE_COLS = 299 # 96 for custom model
    SIZE_CHAN = 3 # use 1 for custom model, 3 for pretrained model

    tol = 1 # % of pixels to saturate (i.e. 5 means keep pixels within 5-95%)

    image_names = [x for x in list(os.scandir(path_data)) if x.is_file() and os.path.splitext(x)[1] == ".png"]
    image_names.sort(key=lambda x: x.name)
    num_images = len(image_names)
    X_data = np.ndarray((num_images, SIZE_ROWS, SIZE_COLS, SIZE_CHAN), dtype=np.uint8)

    for i, image_name in enumerate(image_names):
        img = cv2.imread(os.path.join(path_data, image_name), cv2.IMREAD_GRAYSCALE)

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

    X_datagen = ImageDataGenerator(rescale=1./255)
    X_datagen.fit(X_data, augment=True)
    X_generator = X_datagen.flow(
        X_data,
        batch_size=32,
        shuffle=False)

    predictions = model.predict(X_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    counts_classes = np.bincount(predicted_classes)

    with open(os.path.join(path_output, 'classifications.csv'), 'w', newline='') as f1:
        writer1 = csv.writer(f1)
        writer1.writerow(['Total', class_labels[0], class_labels[1], class_labels[2]])
        writer1.writerow(['', counts_classes[0], counts_classes[1], counts_classes[2]])
        writer1.writerow(['filename', 'classification', 'max class', class_labels[0], class_labels[1], class_labels[2]])
        for i in range(len(predictions)):
            cv2.imwrite(os.path.join(path_output, 'p' + class_labels[predicted_classes[i]] + f'{predictions[i][predicted_classes[i]]:.2f}' + '_' + image_names[i].name), X_data[i,:,:,0])
            writer1.writerow([image_names[i].name, class_labels[predicted_classes[i]], predictions[i][predicted_classes[i]]] + list(predictions[i]))

    # for i in range(len(predictions)):
    #     img = X_data[i]
    #     cv2.imshow('image', img)
    #     cv2.setWindowTitle('image', f'{class_labels[predicted_classes[i]]} {predictions[i][predicted_classes[i]]:.3f}')
    #     cv2.waitKey(0)

if __name__ == "__main__":
    main()