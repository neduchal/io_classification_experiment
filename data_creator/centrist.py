#! /usr/bin/python
# -*- coding: utf-8 -*-


"""@package Centrist 2D wrapper

 Author : Ing. Petr Neduchal
"""


# IMPORT PACKAGES
import numpy as np
import ctypes

# Load Function 
def load() :
    """Function load() loads LBP library centristLibrary.so

			 Return : Library object for calling library functions
    """
    centristlib = ctypes.cdll.LoadLibrary("libCentristLibrary.so")
    return centristlib
 
# LBP radius 1, points 8 algorithm
def centrist(centristlib, npIM) :
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    h = npIM.shape[0]
    w = npIM.shape[1]
    img = (ctypes.c_int32 * (w*h))()
    res = (ctypes.c_int32 * 256)()
    for i in range(h) :
        for j in range(w) :
            img[(w*i) + j] = npIM[i,j]
    centristlib.centristCxx(w,h,ctypes.byref(img), ctypes.byref(res))
    res2 = np.zeros([256, 1], dtype = np.int32)    
    for i in range(256):
      res2[i] = res[i]
    return res2

def centrist_im(centristlib, npIM) :
    """Function calls extern function realTimeLbpImNp

	Params : lbplib - library pointer (see loadLbpLibrary)
                 npIM - NumPy matrix
    """
    h = npIM.shape[0]
    w = npIM.shape[1]
    img = (ctypes.c_int32 * (w*h))()
    res = (ctypes.c_int32 * (w*h))()
    for i in range(h) :
        for j in range(w) :
            img[(w*i) + j] = npIM[i,j]
    centristlib.centristImCxx(w,h,ctypes.byref(img), ctypes.byref(res))
    res2 = np.zeros([h, w], dtype = np.int32)    
    for ih in range(1,h-1):
        for iw in range(1,w-1):
            res2[ih, iw] = res[(w*ih) + iw]
    return res2

