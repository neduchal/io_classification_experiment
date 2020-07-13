import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import confusion_matrix

import vocabulary_helpers
import plot_data
import h5py
from joblib import dump, load


def loadData():
    train_imgs = []
    train_labels = []
    test_imgs = []
    test_labels = []
    with h5py.File("/media/neduchal/data2/datasety/places365_256x256_prepared/miniplaces128_10000/train.h5", "r") as input:
        train_imgs = input["data_x"][()][:10000]
        train_labels = input["data_y_io"][()][:10000]
    with h5py.File("/media/neduchal/data2/datasety/places365_256x256_prepared/miniplaces128_10000/test.h5", "r") as input:
        test_imgs = input["data_x"][()]
        test_labels = input["data_y_io"][()]

    return train_imgs, train_labels, test_imgs, test_labels


train_x, train_y, test_x, test_y = loadData()

print("Training Images: ")

print("Generating vocabulary: ")
(t_f_vocabulary, t_i_vocab) = vocabulary_helpers.generate_vocabulary(test_x)
(f_vocabulary, i_vocab) = vocabulary_helpers.generate_vocabulary(train_x)

print("Generating clusters: ")
n_clusters = 128

kmeans = vocabulary_helpers.generate_clusters(f_vocabulary, n_clusters)

dump(kmeans, 'kmeans30.joblib')
#kmeans = pickle.load(open("bov_pickle_1000.sav", 'rb'))

print("generating features: ")

print("Creating feature vectors for test and train images from vocabulary set.")
train_data = vocabulary_helpers.generate_features(i_vocab, kmeans, n_clusters)
test_data = vocabulary_helpers.generate_features(t_i_vocab, kmeans, n_clusters)

print("Applying SVM classifier.")
# SVM Classifier.
clf = svm.SVC()
fitted = clf.fit(train_data, train_y)
predict = clf.predict(test_data)

print("Actual", "Predicted")
good = 0
for i in range(len(test_y)):
    if test_y[i] == predict[i]:
        good += 1

print(good / len(test_y))

dump(clf, 'svm01.joblib')

# Confusion matrix.
#test_labels = np.asarray(test_labels)
#cnf_matrix = confusion_matrix(predict, test_labels)
# np.set_printoptions(precision=2)

# plot_data.plot_confusion_matrix(cnf_matrix, classes=categories,
#                                title='Confusion matrix')
# plt.show()
