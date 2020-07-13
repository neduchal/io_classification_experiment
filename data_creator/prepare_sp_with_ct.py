import os
import numpy as np
import cv2
import h5py
import random
import description as d
from sklearn import preprocessing
import common_two_phase2 as common
import joblib
from sklearn import svm
from joblib import dump, load
from multiprocessing import Pool, TimeoutError 
import logging
import sys


def process(dataset_name):
    logging.info("{}_  START".format(dataset_name)) 
    print(dataset_name)
    with h5py.File(os.path.join(dataset_name, "test.h5"), "r") as input:
        test_x = input["data_x"][()]
        test_y = input["data_y_io"][()]    

    print("TEST X")
    res = []
    for i in range(len(test_x)):
        temp = (sum(test_x[i]))- 16 > 0     
        res.append(temp == test_y[i])        

    print(100.0*sum(res)/len(test_y))    

    return "OK"

if __name__ == "__main__":
    logging.basicConfig(filename="./second.log", level=logging.DEBUG)
    output_path = "./prepared/"

    process(output_path + "second_phase_hsv_256")
    process(output_path + "second_phase_hsv_128")  
    process(output_path + "second_phase_hsv_64")  
    process(output_path + "second_phase_hsv_32")  
    process(output_path + "second_phase_rgb_256")
    process(output_path + "second_phase_rgb_128")  
    process(output_path + "second_phase_rgb_64")  
    process(output_path + "second_phase_rgb_32") 
    process(output_path + "second_phase_hsv_256_ct")
    process(output_path + "second_phase_hsv_128_ct")  
    process(output_path + "second_phase_hsv_64_ct")  
    process(output_path + "second_phase_hsv_32_ct")  
    process(output_path + "second_phase_rgb_256_ct")
    process(output_path + "second_phase_rgb_128_ct")  
    process(output_path + "second_phase_rgb_64_ct")  
    process(output_path + "second_phase_rgb_32_ct")     
 