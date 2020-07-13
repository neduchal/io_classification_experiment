import tarfile
import os
# import os.path
import numpy as np
import cv2
import h5py
import random
import description as d
from sklearn import preprocessing
import common

dataset_name_base = "wavelet"
directory = "/media/neduchal/data2/datasety/miniplaces/images"
categories_filename = "./categories_io.txt"
output_path = "/media/neduchal/data2/datasety/miniplaces_prepared"
val_path = "./miniplaces/data/val.txt"

print("loading classes")
desc_file = open(categories_filename, "r").read().split("\n")
if desc_file[-1] == "":
    desc_file = desc_file[:-1]

classes = []
classes_nums = []
classes_io = []
for row in desc_file:
    items = row.split(" ")
    classes.append(items[0])
    classes_nums.append(items[1])
    classes_io.append(items[2])

train_directory = os.path.join(directory, "train")
test_directory = os.path.join(directory, "test")
val_directory = os.path.join(directory, "val")

fname_begin_train = len(train_directory)
fname_begin_test = len(test_directory)
fname_begin_val = len(val_directory)    

val_names = common.get_all_files(val_directory)
classes_val = open(val_path).read().split("\n")
train_names = common.get_all_files(train_directory)
random.shuffle(train_names)
test_names = train_names[0:int(0.1 * len(train_names))]
train_names = train_names[int(0.1 * len(train_names)):]

dataset_name = dataset_name_base
print("Creating dataset " + dataset_name)
output_directory = os.path.join(output_path, dataset_name)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
print("VAL")
val_data_x, val_data_y, val_data_y_io = common.process_wave(val_names, classes_val, classes_io, classes_nums, fname_begin_val, True)
common.save_to_h5(os.path.join(output_directory, 'val.h5'), val_data_x, val_data_y, val_data_y_io)
print("TRAIN")
train_data_x, train_data_y, train_data_y_io = common.process_wave(train_names, classes, classes_io, classes_nums, fname_begin_train, False)
common.save_to_h5(os.path.join(output_directory, 'train.h5'), train_data_x, train_data_y, train_data_y_io)
print("TEST")
test_data_x, test_data_y, test_data_y_io = common.process_wave(test_names, classes, classes_io, classes_nums, fname_begin_train, False)
common.save_to_h5(os.path.join(output_directory, 'test.h5'), test_data_x, test_data_y, test_data_y_io)
print("DONE")