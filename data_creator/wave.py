import pywt
import numpy as np
import functools
import operator
from scipy import ndimage, misc

def wave(im):
    w = pywt.wavedec2(im, "bior3.5", level=2)
    h, s, sh= pywt.ravel_coeffs(w)
    e = []
    h0 = h[s[0]]
    h0 = np.array(h0).reshape(sh[0][0],sh[0][1])
    e0 = (1/(sh[0][0]*sh[0][1]))*sum(ndimage.laplace(h0).ravel()**2)

    e.append(e0)

    for key in s[1].keys():
        hi = h[s[1][key]]
        e.append((1/len(hi))*sum(hi**2))

    for key in s[2].keys():
        hi = h[s[2][key]]
        e.append((1/len(hi))*sum(hi**2))    
        
    return e 

if __name__ == "__main__":
    wave(np.ones([128,128]))

