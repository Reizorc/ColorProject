import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
 
def describe(img,nbpalier):
    n = nbpalier
    d = 256 / n
    img = img/d
    img = img.astype(int)
    img = img*d
    imhist = img[:,:,0]+n*img[:,:,1]+n*n*img[:,:,2]
    (hist, _)= np.histogram(imhist.ravel(),bins =pow(n,3))
    hist = hist.astype("float")
    hist /= (hist.sum()+1e-7)
    return hist