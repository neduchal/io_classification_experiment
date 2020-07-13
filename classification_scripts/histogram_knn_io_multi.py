from multiprocessing import Pool, TimeoutError 
import os
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import vocabulary_helpers
import plot_data
import h5py
from joblib import dump, load
import pickle

def process(directory, dataset_name):
    print(dataset_name, "START")
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

    print(dataset_name, "Applying KNN classifier.")
    # KNN Classifier.
    neigh = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
    start_train = time.time()
    fitted = neigh.fit(train_x, train_y)
    end_train = time.time()
    predict = neigh.predict(test_x)
    end_test = time.time()
    good = 0
    for i in range(len(test_y)):
        if test_y[i] == predict[i]:
            good += 1
    dump(neigh, "knn_"+ dataset_name +".joblib")  
    accuracy =  good / len(test_y)     
    test_y_0 = np.sum(test_y == 0)
    test_y_1 = np.sum(test_y == 1)
  
    print(dataset_name, "Accuracy:", good / len(test_y), "Train time", end_train-start_train)
    predict_list = list(predict)
    test_y_list = list(test_y)
    predict_int = [int(k) for k in predict_list]
    test_y_int = [int(k) for k in test_y_list]
    result = {"accuracy": accuracy, "good": int(good), "test_count":len(test_y), "test_y_0": int(test_y_0), "test_y_1": int(test_y_1), "predict": predict_int, "test_values": test_y_int}
    return [dataset_name, result]


if __name__ == "__main__":
    directory = "/media/neduchal/data2/datasety/miniplaces_prepared/"
    #types = ["rgb", "hsv"]
    types = [ "luv", "ohta"]
    #types = ["rgb"]
    #hist_settings = [(2,2,4), (2,2,8), (3,3,4), (3,3,8), (4,4,4), (4,4,8)]
    #hist_settings = [ (2,2,8), (2,2,16), (3,3,4), (3,3,8), (4,4,4), (4,4,8)]
    #hist_settings = [ (2,2,8), (2,2,16), (3,3,4)] 
    hist_settings = [(5,5,16)]
    # GENERATED: (5,5,4), (5,5,8)
    dataset_name_base = "io_hist__"
    result_file = "results_knn_5_5_X.pickle"

    dataset_names = [dataset_name_base+item1+"_"+str(item2[0])+"_"+str(item2[1])+"_"+str(item2[2]) for item1 in types for item2 in hist_settings]
    
    pool = Pool(processes=len(types)*len(hist_settings))
    multiple_results = [pool.apply_async(process, (directory, d_name)) for d_name in dataset_names[0:6]]
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
        pickle_results[res[0]] = res[1]
    pickle.dump(pickle_results, f)
    f.close()

# Confusion matrix.
#test_labels = np.asarray(test_labels)
#cnf_matrix = confusion_matrix(predict, test_labels)
# np.set_printoptions(precision=2)

# plot_data.plot_confusion_matrix(cnf_matrix, classes=categories,
#                                title='Confusion matrix')
# plt.show()
