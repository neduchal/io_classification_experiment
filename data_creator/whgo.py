import cv2
import numpy as np
import math

def toBins(im, bins):
    output = np.zeros([im.shape[0], im.shape[1]], dtype=np.uint8)
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            output[y,x] = math.ceil((im[y,x]/(2*np.pi)) * bins)
    return output

def whgo(im, bins):
    derivX = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    derivY = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)  
    orien = cv2.phase(derivX, derivY, angleInDegrees=False)
    mag = cv2.magnitude(derivX, derivY)    
    orien_bins = toBins(orien, bins)
    M = sum(sum(mag))
    W = []
    H = []    
    for b in range(bins):
        W.append(sum(mag[orien_bins == b])/M)
        H.append(W[-1]* (sum(orien_bins[orien_bins == b])/(im.shape[0]*im.shape[1])))
    return H





