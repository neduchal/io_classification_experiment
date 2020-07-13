import tarfile
import os
# import os.path
import numpy as np
import cv2
import h5py
import random

dataset_name = "miniplaces128"

print("Creating dataset " + dataset_name)
print()


directory = "/media/neduchal/data2/datasety/miniplaces/images"


train_directory = os.path.join(directory, "train")
test_directory = os.path.join(directory, "test")
val_directory = os.path.join(directory, "val")

fname_begin_train = len(train_directory)
fname_begin_test = len(test_directory)
fname_begin_val = len(val_directory)

output_directory = os.path.join("/media/neduchal/data2/datasety/places365_256x256_prepared", dataset_name)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

print("VAL")

print("loading val files")

val_names = []

for root, dirs, files in os.walk(val_directory):
    for name in files:
        val_names.append(os.path.join(root, name))

classes = open("/media/neduchal/data2/datasety/miniplaces/miniplaces/data/val.txt").read().split("\n")

print("loading val data")

val_data_x = []
val_data_y = []

for i, filename in enumerate(val_names):
    im = cv2.imread(filename)
    index = classes.index(filename[fname_begin_val:-13])
    val_data_x.append(im)
    val_data_y.append(int(classes[i].split(" ")[1]))

print("saving val data")

val_f = h5py.File(os.path.join(output_directory, 'val.h5'), 'w')

val_f["data_x"] = val_data_x
val_f["data_y"] = val_data_y

val_f.flush()
val_f.close()

del(val_data_x)
del(val_data_y)

print("TEST")

print("loading test files")

test_names = []

for root, dirs, files in os.walk(test_directory):
    for name in files:
        test_names.append(os.path.join(root, name))

print("loading test data")

test_data_x = []

for i, filename in enumerate(test_names):
    im = cv2.imread(filename)
    index = classes.index(filename[fname_begin_test:-13])
    test_data_x.append(im)

print("saving test tarfile")

test_f = h5py.File(os.path.join(output_directory, 'test.h5'), 'w')

test_f["data_x"] = test_data_x

test_f.flush()
test_f.close()

del(test_data_x)

print("TRAIN")

desc_file = open("/media/neduchal/data2/datasety/miniplaces/miniplaces/data/categories.txt",
                 "r").read().split("\n")

print("loading classes")
if desc_file[-1] == "":
    desc_file = desc_file[:-1]

classes = []
classes_nums = []
#classes_io = []

for row in desc_file:
    items = row.split(" ")
    classes.append(items[0])
    classes_nums.append(items[1])
    # classes_io.append(items[2])

print("loading train data")

train_names = []

for root, dirs, files in os.walk(train_directory):
    for name in files:
        train_names.append(os.path.join(root, name))

random.shuffle(train_names)

train_data_x = []
train_data_y = []
#train_data_y_io = []


print("processing train data")

for filename in train_names:
    im = cv2.imread(filename)
    index = classes.index(filename[fname_begin_train:-13])
    train_data_x.append(im)
    train_data_y.append(int(classes_nums[index]))
    # train_data_y_io.append(classes_io[index])

print("saving train data")

train_f = h5py.File(os.path.join(output_directory, 'train.h5'), 'w')

train_f["data_x"] = train_data_x
train_f["data_y"] = train_data_y
#train_f["data_y_io"] = train_data_y_io

train_f.flush()
train_f.close()

del(train_data_x)
del(train_data_y)
# del(train_data_y_io)

print("DONE")
