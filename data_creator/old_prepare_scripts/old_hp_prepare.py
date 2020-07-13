import h5py
import cv2
import random

files = open("filelist.txt", "r").read().split("\n")

random.shuffle(files)


train_count = int(3/4.0 * len(files))
test_count = len(files) - train_count

train_data_x = []
train_data_y = []

for line in files[0:train_count]:
    line_arr = line.split(",")
    if len(line_arr) < 2:
        continue    
    img = cv2.imread(line_arr[0], 0)
    img = cv2.resize(img, (80, 60))
    train_data_x.append(img)
    train_data_y.append(int(line_arr[1]))

test_data_x = []
test_data_y = []

for line in files[train_count:]:
    line_arr = line.split(",")
    if len(line_arr) < 2:
        continue
    img = cv2.imread(line_arr[0], 0)
    img = cv2.resize(img, (80, 60))
    test_data_x.append(img)
    test_data_y.append(int(line_arr[1]))


train_f = h5py.File('train_data_small_v2.h5','w')

train_f["data_x"] = train_data_x
train_f["data_y"] = train_data_y

train_f.flush()
train_f.close()

test_f = h5py.File('test_data_small_v2.h5','w')

test_f["data_x"] = test_data_x
test_f["data_y"] = test_data_y

test_f.flush()
test_f.close()

