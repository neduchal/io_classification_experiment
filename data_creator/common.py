import os
import cv2
import description as d
from sklearn import preprocessing
import h5py
import centrist
import gbp
import numpy as np
import wave
import whgo

def get_all_files(directory):
    #for root, dirs, files in os.walk(directory):
    #    filenames = [os.path.join(root, name) for name in files]
    filenames = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            filenames.append(os.path.join(root, name))    
    return filenames

def process_whgo(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, hs=8):
    data_x = []
    data_y = []
    data_y_io = []

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        img = cv2.imread(filename, 0)
        im = img.copy()
        h1 = whgo.whgo(im[0:im.shape[0]//2, 0:im.shape[1]//2], hs)
        h2 = whgo.whgo(im[0:im.shape[0]//2, im.shape[1]//2:im.shape[1]], hs)
        h3 = whgo.whgo(im[im.shape[0]//2:im.shape[0], 0:im.shape[1]//2], hs)
        h4 = whgo.whgo(im[im.shape[0]//2:im.shape[1], im.shape[1]//2:im.shape[1]], hs)    
        h = np.concatenate((h1,h2,h3,h4))                    
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io         


def process_wave(filenames, classes, classes_io, classes_nums, fname_base_len, val=False):
    data_x = []
    data_y = []
    data_y_io = []

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        img = cv2.imread(filename, 0)
        im = img.copy()
        h = wave.wave(im)
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io         

def process_gbp(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, hist_settings=(3,3,8)):
    data_x = []
    data_y = []
    data_y_io = []
    hs = hist_settings

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        img = cv2.imread(filename, 0)
        im = img.copy()
        im = gbp.gbp(im)
        h = d.spatial_histogram(im, hs[0], hs[1], hs[2])
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io     

def process_gbp_multiscale(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, hist_settings=8):
    data_x = []
    data_y = []
    data_y_io = []
    hs = hist_settings

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        img = cv2.imread(filename, 0)
        im = img.copy()
        im2 = cv2.resize(im, dsize=(im.shape[1]//2, im.shape[0]//2))
        im3 = cv2.resize(im, dsize=(im.shape[1]//4, im.shape[0]//4))        
        im = gbp.gbp(im)
        im2 = gbp.gbp( im2)
        im3 = gbp.gbp( im3)
        h1 = d.spatial_histogram_bw(im3, 1, 1 , hs)
        h2 = d.spatial_histogram_bw(im2, 2, 2, hs)
        h3 = d.spatial_histogram_bw(im, 4, 4, hs)
        h = np.concatenate((h1, h2, h3))
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io           

def process_centrist_io(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, hist_settings=(3,3,8)):
    cl = centrist.load()
    data_x = []
    data_y = []
    data_y_io = []
    hs = hist_settings

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        img = cv2.imread(filename)
        im = img.copy()
        im[:,:,0] = centrist.centrist_im(cl, im[:,:,0])
        im[:,:,1] = centrist.centrist_im(cl, im[:,:,1])
        im[:,:,2] = centrist.centrist_im(cl, im[:,:,2])
        h = d.spatial_histogram(im, hs[0], hs[1], hs[2])
        
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io  
    
      

def process_centrist_io_multiscale(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, hist_settings=4):
    cl = centrist.load()
    data_x = []
    data_y = []
    data_y_io = []
    hs = hist_settings

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        img = cv2.imread(filename)
        im = img.copy()
        im[:,:,0] = centrist.centrist_im(cl, im[:,:,0])
        im[:,:,1] = centrist.centrist_im(cl, im[:,:,1])
        im[:,:,2] = centrist.centrist_im(cl, im[:,:,2])
        h1 = d.spatial_histogram(im, 1, 1 , hs)
        h2 = d.spatial_histogram(im, 2, 2, hs)
        h3 = d.spatial_histogram(im, 4, 4, hs)
        h = np.concatenate((h1, h2, h3))
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io    

def process_centrist_io_multiscale_bw(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, hist_settings=4):
    cl = centrist.load()
    data_x = []
    data_y = []
    data_y_io = []
    hs = hist_settings

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        img = cv2.imread(filename, 0)      
        im = img.copy()       
        im2 = cv2.resize(im, dsize=(im.shape[1]//2, im.shape[0]//2))
        im3 = cv2.resize(im, dsize=(im.shape[1]//4, im.shape[0]//4))        
        im = centrist.centrist_im(cl, im)
        im2 = centrist.centrist_im(cl, im2)
        im3 = centrist.centrist_im(cl, im3)
        h1 = d.spatial_histogram_bw(im3, 1, 1 , hs)
        h2 = d.spatial_histogram_bw(im2, 2, 2, hs)
        h3 = d.spatial_histogram_bw(im, 4, 4, hs)
        h = np.concatenate((h1, h2, h3))
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io 

def process_hist_io(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, preproc="rgb", hist_settings=(3,3,8)):
    data_x = []
    data_y = []
    data_y_io = []
    hs = hist_settings

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        im = cv2.imread(filename)
        if preproc == "rgb":
            h = d.spatial_histogram(im, hs[0], hs[1], hs[2])
        elif preproc == "hsv":
            h = d.spatial_histogram_hsv(im, hs[0], hs[1], hs[2])
        elif preproc == "luv":
            h = d.spatial_histogram_luv(im, hs[0], hs[1], hs[2])
        elif preproc == "ohta":
            h = d.spatial_histogram_ohta(im, hs[0], hs[1], hs[2])
        else:
            print("Unknown preprocessing method")
            exit(1)
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io


def process_hist_io_multiscale(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, preproc="rgb", hist_settings=(3,3,8)):
    data_x = []
    data_y = []
    data_y_io = []
    hs = hist_settings

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        im = cv2.imread(filename)
        im2 = cv2.resize(im, dsize=(im.shape[1]//2, im.shape[0]//2))
        im3 = cv2.resize(im, dsize=(im.shape[1]//4, im.shape[0]//4)) 
        if preproc == "rgb":
            h1 = d.spatial_histogram(im3, 1, 1, hs)
            h2 = d.spatial_histogram(im2, 2, 2, hs)
            h3 = d.spatial_histogram(im, 4, 4, hs)
        elif preproc == "hsv":
            h1 = d.spatial_histogram_hsv(im3, 1, 1, hs)
            h2 = d.spatial_histogram_hsv(im2, 2, 2, hs)
            h3 = d.spatial_histogram_hsv(im, 4, 4, hs)
        elif preproc == "luv":
            h1 = d.spatial_histogram_luv(im3, 1, 1, hs)
            h2 = d.spatial_histogram_luv(im2, 2, 2, hs)
            h3 = d.spatial_histogram_luv(im, 4, 4, hs)
        elif preproc == "ohta":
            h1 = d.spatial_histogram_ohta(im3, 1, 1, hs)
            h2 = d.spatial_histogram_ohta(im2, 2, 2, hs)
            h3 = d.spatial_histogram_ohta(im, 4, 4, hs)
        else:
            print("Unknown preprocessing method")
            exit(1)
        h = np.concatenate((h1, h2, h3))  
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io

def process_nbhs_io(filenames, classes, classes_io, classes_nums, fname_base_len, val=False, hist_settings=(3,3,8)):
    data_x = []
    data_y = []
    data_y_io = []
    hs = hist_settings

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        im = cv2.imread(filename)
        h = d.spatial_nbhs(im, hs[0], hs[1])
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io

def process_nbhs_io_multiscale(filenames, classes, classes_io, classes_nums, fname_base_len, val=False):
    data_x = []
    data_y = []
    data_y_io = []

    for i, filename in enumerate(filenames):
        printProgressBar(i+1, len(filenames))
        im = cv2.imread(filename)
        im2 = cv2.resize(im, dsize=(im.shape[1]//2, im.shape[0]//2))
        im3 = cv2.resize(im, dsize=(im.shape[1]//4, im.shape[0]//4))         
        h1 = d.spatial_nbhs(im3, 1, 1)
        h2 = d.spatial_nbhs(im2, 2, 2)
        h3 = d.spatial_nbhs(im, 4, 4)      
        h = np.concatenate((h1, h2, h3))                    
        index = classes.index(filename[fname_base_len:-13])
        data_x.append(h)
        if val:
            inout = int(classes_io[int(classes[i].split(" ")[1])])
            data_y.append(int(classes[i].split(" ")[1]))
            data_y_io.append(inout)
        else:
            data_y.append(int(classes_nums[index]))
            data_y_io.append(int(classes_io[index]))
    data_x_scaled = preprocessing.scale(data_x)
    return data_x_scaled, data_y, data_y_io    

def save_to_h5(filename, data_x, data_y, data_y_io):
    with h5py.File(filename, "w") as f:
        f["data_x"] = data_x
        f["data_y"] = data_y
        f["data_y_io"] = data_y_io
        f.flush()    

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()