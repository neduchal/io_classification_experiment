import numpy as np
from scipy.signal import convolve


def gbp(img):
    g1 = np.array([[-1,0,1]])
    g2 = np.array([[-1],[0],[1]])
    g3 = np.array([[0,0,1],[0,0,0],[-1,0,0]])
    g4 = np.array([[-1,0,0],[0,0,0],[0,0,1]]) 


    rg1 = convolve(img, g1, mode="same")
    rg2 = convolve(img, g2, mode="same")
    rg3 = convolve(img, g3, mode="same")
    rg4 = convolve(img, g4, mode="same")

    s1 = (np.abs(rg1) - np.abs(rg4)) >= 0
    s2 = (np.abs(rg3) - np.abs(rg4)) >= 0
    s3 = (np.abs(rg1) - np.abs(rg2)) >= 0
    s7 = (np.abs(rg1)) >= 0    
    s6 = (np.abs(rg2)) >= 0    
    s5 = (np.abs(rg3)) >= 0    
    s4 = (np.abs(rg4)) >= 0    
       
    gbp_im = s1*(2**0) + s2*(2**1) + s3*(2**2) + s4*(2**3) + s5*(2**4) + s6*(2**5) + s7*(2**6)
    return gbp_im



