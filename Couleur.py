from skimage import data
import numpy as np
from PIL import Image as pil
import matplotlib.pyplot as plt
from math import floor





def boucle(img):
    for i in range(322):
        for j in range(480):
            img[i,j,1] = floor(img[i,j,1])
            img[i,j,2] = floor(img[i,j,2])
            img[i,j,0] = floor(img[i,j,0])
    return img





def describe(img,nbpalier):
    n = nbpalier
    d = 256 / n
    i = img.dot(1/d)
    imageq = boucle(i)
    image = imageq.dot(d)
    imagef = img[:,:,0] + img[:,:,1].dot(n) + img[:,:,2].dot(n*n)
    (hist, _) = np.histogram(imagef,
        bins='auto')
    return (hist,imagef)


image = pil.open("res/Tomate/Tamotoes0072.png")
img=np.array(image)
(hist,imagef) = describe(img,4)




plt.hist(hist)
plt.show()


