import tarfile
import os
import numpy as np
import cv2
import h5py
import random

# Parameters
test_count = 1000
dataset_name = "basic_io"
directory = "/media/neduchal/data2/datasety/miniplaces/images"
categories_filename = "./categories_io.txt"
output_path = "/media/neduchal/data2/datasety/places365_256x256_prepared"
val_path = "/media/neduchal/data2/datasety/miniplaces/miniplaces/data/val.txt"

print("Creating dataset " + dataset_name)
print()

desc_file = open(categories_filename, "r").read().split("\n")

print("loading classes")
if desc_file[-1] == "":
    desc_file = desc_file[:-1]

# Classes loading
classes = []
classes_nums = []
classes_io = []
for row in desc_file:
    items = row.split(" ")
    classes.append(items[0])
    classes_nums.append(items[1])
    classes_io.append(items[2])

# Get directories
train_directory = os.path.join(directory, "train")
test_directory = os.path.join(directory, "test")
val_directory = os.path.join(directory, "val")

# Get number of chars of directory
fname_begin_train = len(train_directory)
fname_begin_test = len(test_directory)
fname_begin_val = len(val_directory)

# Set output directory and creates it if it is not exists
output_directory = os.path.join(output_path, dataset_name)
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

print("VAL")
print("loading val files")

val_names = []
for root, dirs, files in os.walk(val_directory):
    for name in files:
        val_names.append(os.path.join(root, name))

classes_val = open(val_path).read().split("\n")

print("loading val data")

val_data_x = []
val_data_y = []
val_data_y_io = []
for i, filename in enumerate(val_names):
    print(int(100*(i/len(val_names))))
    im = cv2.imread(filename)
    index = classes_val.index(filename[fname_begin_val:-13])
    val_data_x.append(im)
    inout = int(classes_io[int(classes_val[i].split(" ")[1])])
    val_data_y.append(int(classes_val[i].split(" ")[1]))
    val_data_y_io.append(inout)

print("saving val data")

val_f = h5py.File(os.path.join(output_directory, 'val.h5'), 'w')
val_f["data_x"] = val_data_x
val_f["data_y"] = val_data_y
val_f["data_y_io"] = val_data_y_io

val_f.flush()
val_f.close()

del(val_data_x)
del(val_data_y)

print("TRAIN")

desc_file = open("./categories_io.txt",
                 "r").read().split("\n")

print("loading classes")
if desc_file[-1] == "":
    desc_file = desc_file[:-1]

classes = []
classes_nums = []
classes_io = []
classes_count = []

for row in desc_file:
    items = row.split(" ")
    classes.append(items[0])
    classes_nums.append(items[1])
    classes_io.append(items[2])
    classes_count.append(0)

print("loading train data")

train_names = []

for root, dirs, files in os.walk(train_directory):
    for name in files:
        train_names.append(os.path.join(root, name))

random.shuffle(train_names)

test_names = train_names[0:int(0.1 * len(train_names))]
train_names = train_names[int(0.1 * len(train_names)):]
print(len(test_names), train_names)

train_data_x = []
train_data_y = []
train_data_y_io = []


print("processing train data")

for i, filename in enumerate(train_names):
    if (i % 5000) == 0:
        print(int(100*(i/len(train_names))))
    index = classes.index(filename[fname_begin_train:-13])
    if classes_count[index] >= 100:
        continue
    classes_count[index] += 1
    im = cv2.imread(filename)
    train_data_x.append(im)
    train_data_y.append(int(classes_nums[index]))
    train_data_y_io.append(int(classes_io[index]))

print("saving train data")

train_f = h5py.File(os.path.join(output_directory, 'train.h5'), 'w')

train_f["data_x"] = train_data_x
train_f["data_y"] = train_data_y
train_f["data_y_io"] = train_data_y_io

train_f.flush()
train_f.close()

del(train_data_x)
del(train_data_y)
del(train_data_y_io)


print("TEST")

print("loading test data")

test_data_x = []
test_data_y = []
test_data_y_io = []

for i, filename in enumerate(test_names):
    if (i % 100) == 0:
        print(int(100*(i/test_count)))
    if i >= test_count:
        break
    im = cv2.imread(filename)
    index = classes.index(filename[fname_begin_train:-13])
    test_data_x.append(im)
    test_data_y.append(int(classes_nums[index]))
    test_data_y_io.append(int(classes_io[index]))


print("saving test h5 file")

test_f = h5py.File(os.path.join(output_directory, 'test.h5'), 'w')

test_f["data_x"] = test_data_x
test_f["data_y"] = test_data_y
test_f["data_y_io"] = test_data_y_io

test_f.flush()
test_f.close()

del(test_data_x)
del(test_data_y)
del(test_data_y_io)

print("DONE")
