import tarfile
import os
# import os.path
import numpy as np
import cv2
import h5py
import random

dataset_name = "full128"

print("TRAIN")

desc_file = open("/media/neduchal/data2/datasety/places365/places_devkit/categories_places365_io.txt",
                 "r").read().split("\n")

print("loading classes")
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

print("loading train tarfile")

directory = "/media/neduchal/data2/datasety/places365_256x256"


train_tarfilename = os.path.join(directory, "train_256_places365standard.tar")
test_tarfilename = os.path.join(directory, "test_256.tar")
val_tarfilename = os.path.join(directory, "val_256.tar")

output_directory = os.path.join("/media/neduchal/data2/datasety/places365_256x256_prepared", dataset_name)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

print("loading train data")

train_tarfile = tarfile.open(train_tarfilename, mode='r')
train_names = train_tarfile.getnames()
random.shuffle(train_names)

train_data_x = []
train_data_y = []
train_data_y_io = []

print("processing train data")

for filename in train_names:
    member = train_tarfile.getmember(filename)
    if member is None:
        continue
    if member.isdir():
        continue
    img_data = train_tarfile.extractfile(member).read()
    img_np_array = np.frombuffer(img_data, dtype=np.uint8)
    im = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)
    im_resized = cv2.resize(im, (128, 128))

    index = classes.index(member.name[8:-13])
    train_data_x.append(im_resized)
    train_data_y.append(classes_nums[index])
    train_data_y_io.append(classes_io[index])

print("saving train data")

train_f = h5py.File(os.path.join(output_directory, 'train.h5'), 'w')

train_f["data_x"] = train_data_x
train_f["data_y"] = train_data_y
train_f["data_y_io"] = train_data_y_io

train_f.flush()
train_f.close()
train_tarfile.close()

del(train_data_x)
del(train_data_y)
del(train_data_y_io)

print("VAL")

print("loading val tarfile")

val_tarfile = tarfile.open(val_tarfilename, mode='r')
val_names = val_tarfile.getnames()

classes = open("/media/neduchal/data2/datasety/places365/places_devkit/places365_val.txt").read().split("\n")

print("loading val data")

val_data_x = []
val_data_y = []

for i, name in enumerate(val_names):
    img_data = val_tarfile.extractfile(member).read()
    img_np_array = np.frombuffer(img_data, dtype=np.uint8)
    im = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)
    im_resized = cv2.resize(im, (128, 128))
    val_data_x.append(im_resized)
    val_data_y.append(classes[i].split(" ")[1])

print("saving val data")

val_f = h5py.File(os.path.join(output_directory, 'val.h5'), 'w')

val_f["data_x"] = val_data_x
val_f["data_y"] = val_data_y

val_f.flush()
val_f.close()
val_tarfile.close()

del(val_data_x)
del(val_data_y)

print("TEST")

print("loading test tarfile")

test_tarfile = tarfile.open(test_tarfilename, mode='r')
test_names = test_tarfile.getnames()

print("loading test data")

test_data_x = []

for i, name in enumerate(test_names):
    img_data = test_tarfile.extractfile(member).read()
    img_np_array = np.frombuffer(img_data, dtype=np.uint8)
    im = cv2.imdecode(img_np_array, cv2.IMREAD_COLOR)
    im_resized = cv2.resize(im, (128, 128))
    test_data_x.append(im_resized)

print("saving test tarfile")

test_f = h5py.File(os.path.join(output_directory, 'test.h5'), 'w')

test_f["data_x"] = test_data_x

test_f.flush()
test_f.close()
test_tarfile.close()

del(test_data_x)

print("DONE")
