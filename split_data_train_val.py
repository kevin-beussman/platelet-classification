import os
import random

def main():
    val_pct = 0.2 # 20%

    path_data = "C:/Users/kevin/git-workspace/tf-platelets/Data/"
    path_train_data = "C:/Users/kevin/git-workspace/tf-platelets/Data/Training/"
    path_val_data = "C:/Users/kevin/git-workspace/tf-platelets/Data/Validation/"

    data_labels = os.listdir(path_data)

    if not os.path.exists(path_train_data):
        os.mkdir(path_train_data)
        for label in data_labels:
            if not os.path.exists(os.path.join(path_train_data, label)):
                os.mkdir(os.path.join(path_train_data, label))
    if not os.path.exists(path_val_data):
        os.mkdir(path_val_data)
        for label in data_labels:
            if not os.path.exists(os.path.join(path_val_data, label)):
                os.mkdir(os.path.join(path_val_data, label))

    num_samples = {}
    for j, label in enumerate(data_labels):
        if label == "Training" or label == "Validation":
            continue
        num_samples[label] = len([1 for x in list(os.scandir(os.path.join(
            path_data, label))) if x.is_file() and os.path.splitext(x)[1] == ".png"])

    for j, label in enumerate(data_labels):
        if label == "Training" or label == "Validation":
            continue
        # val_ind = np.random.choice(num_samples[label], int(num_samples[label]*val_pct))
        val_ind = random.sample(range(num_samples[label]), int(num_samples[label]*val_pct))
        val_ind.sort()
        print(val_ind[0:10])
        print(f'({num_samples[label]},{int(num_samples[label]*val_pct)})')
        image_names = [x.name for x in list(os.scandir(os.path.join(
            path_data, label))) if x.is_file() and os.path.splitext(x)[1] == ".png"]

        for i, image_name in enumerate(image_names):
            if val_ind and i == val_ind[0]:
                val_ind.pop(0)
                os.rename(os.path.join(path_data, label, image_name), os.path.join(path_val_data, label, image_name))
            else:
                os.rename(os.path.join(path_data, label, image_name), os.path.join(path_train_data, label, image_name))

        os.rmdir(os.path.join(path_data, label))


if __name__ == "__main__":
    main()