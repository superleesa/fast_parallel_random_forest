__author__ = "Satoshi Kashima"

import os
from sklearn.datasets import load_svmlight_files
import numpy as np
from sklearn.model_selection import train_test_split


def load_files(files_to_load, num_of_features, change_range=False):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_dir = os.getcwd()
    data_dir = 'url_svmlight'  # might need to change this depending on the OS

    # files = map(lambda x: os.path.join(root_dir, data_dir, x), os.listdir(os.path.join(root_dir, data_dir))[: files_to_load])
    files = map(lambda x: os.path.join(current_dir, data_dir, x),
                os.listdir(os.path.join(current_dir, data_dir))[: files_to_load])

    return_array = load_svmlight_files(files)
    features = []
    labels = []

    for each_value in return_array:
        if type(each_value) != np.ndarray:
            features.extend(each_value[:, :num_of_features].toarray().tolist())
        else:
            labels.extend(each_value.tolist())

    features = np.array(features)
    labels = np.array(labels)

    if change_range:
        labels = np.array(list(map(lambda x: int(x>0), labels)))

    return train_test_split(features, labels, train_size=0.80)