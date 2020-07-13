from multiprocessing import Pool, TimeoutError
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import h5py
from joblib import dump, load
import pickle
import logging
from sklearn.decomposition import PCA


def process(directory, dataset_name, cl_type, joblibs_dir="./", n_components=128):
    logging.info("{} START".format(dataset_name))
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    dataset_dir = os.path.join(directory, dataset_name)
    with h5py.File(os.path.join(dataset_dir, "train.h5"), "r") as input:
        train_x = input["data_x"][()]
        train_y = input["data_y_io"][()]
    with h5py.File(os.path.join(dataset_dir, "test.h5"), "r") as input:
        test_x = input["data_x"][()]
        test_y = input["data_y_io"][()]
    if len(train_x[0]) < n_components:
        return None, None

    train_x = preprocessing.scale(train_x)
    test_x = preprocessing.scale(test_x)
    pca = PCA(n_components=n_components)
    pca.fit(train_x)
    dump(pca, os.path.join(joblibs_dir, "pca" + str(n_components) + "_" + dataset_name + ".joblib"))
    pca_train_x = pca.transform(train_x)
    start_transform = time.time()
    pca_test_x = pca.transform(test_x)
    end_transform = time.time()
    logging.info("Transform time for one example: {} ".format((end_transform-start_transform)/10000.0))

    joblibname = cl_type + "_" + dataset_name + "pca" + str(n_components) + ".joblib"
    clf = None
    predict = []
    if (cl_type == "svm"):
        clf = svm.SVC(cache_size=4096)
    elif (cl_type == "lsvm"):
        clf = svm.LinearSVC(max_iter=10000)
    elif (cl_type == "naive"):
        clf = GaussianNB()
    elif (cl_type == "sgd"):
        clf = SGDClassifier()
    elif (cl_type == "tree"):
        clf = tree.DecisionTreeClassifier()
    elif (cl_type == "knn"):
        clf = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto',
                                   leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    else:
        return None, None
    print(dataset_name, "Applying " + cl_type + " classifier.")
    logging.info("{} Applying {} classifier".format(dataset_name, cl_type))
    start_train = time.time()
    fitted = clf.fit(pca_train_x, train_y)
    end_train = time.time()
    predict = clf.predict(pca_test_x)
    end_test = time.time()

    good = 0
    for i in range(len(test_y)):
        if test_y[i] == predict[i]:
            good += 1
    dump(clf, os.path.join(joblibs_dir, joblibname))
    accuracy = good / len(test_y)
    test_y_0 = np.sum(test_y == 0)
    test_y_1 = np.sum(test_y == 1)

    print(dataset_name, "Accuracy:", good / len(test_y), "Train time", end_train-start_train)
    predict_list = list(predict)
    test_y_list = list(test_y)
    predict_int = [int(k) for k in predict_list]
    test_y_int = [int(k) for k in test_y_list]
    logging.info("{} Accuracy: {}".format(dataset_name, accuracy))
    result = {"accuracy": accuracy, "good": int(good), "test_count": len(test_y), "test_y_0": int(
        test_y_0), "test_y_1": int(test_y_1), "predict": predict_int, "test_values": test_y_int}
    return [cl_type + "_" + dataset_name, result]


if __name__ == "__main__":
    logging.basicConfig(filename="/storage/plzen1/home/neduchal/projekty/inout/log/" +
                        str(sys.argv[3]) + '.log', level=logging.DEBUG)
    directory = "/storage/plzen1/home/neduchal/projekty/inout/data/"
    joblibs_dir = "/storage/plzen1/home/neduchal/projekty/inout/joblibs/"
    result_file = "/storage/plzen1/home/neduchal/projekty/inout/results/results_"+str(sys.argv[3])+".pickle"
    logging.info("Directory: {}".format(directory))
    logging.info("Joblibs directory: {}".format(joblibs_dir))
    logging.info("Result file: {}".format(result_file))
    logging.info("{}".format(sys.argv))
    max_processes_at_once = 8
    dname = sys.argv[1]
    pca_settings = sys.argv[4].split(";")
    cl_types = sys.argv[2].split(";")

    for cl_type in cl_types:
        for pc in pca_settings:
            i_pc = int(pc)
            pool = Pool(processes=max_processes_at_once)
            multiple_results = [pool.apply_async(process, (directory, dname, cl_type, joblibs_dir, i_pc))
                                for pc in pca_settings]
            results = [res.get(timeout=None) for res in multiple_results]
            if not os.path.exists(result_file):
                pickle_results = {}
                f = open(result_file, "wb")
            else:
                f = open(result_file, "rb")
                pickle_results = pickle.load(f)
                f.close()
                f = open(result_file, "wb")
            for res in results:
                if res == None:
                    continue
                pickle_results[res[0]] = res[1]
            pickle.dump(pickle_results, f)
            f.close()
