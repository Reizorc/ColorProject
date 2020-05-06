# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:57:21 2020

@author: Win7
"""

import tkinter as tk
#from PIL import Image as pil 
#from PIL import ImageTk as pol
import cv2 #hsv #image_hsv=cv2.cvtColor(imageBis, cv2.COLOR_RGB2HSV())
import colorsys #hsv #colorsys.rgb_to_hsv(0.2, 0.4, 0.4)
import matplotlib.pyplot as plt
from skimage import data
from skimage.measure import regionprops
from skimage.color import rgb2hsv
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from skimage import color
from skimage import io

from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.color import label2rgb

from skimage.morphology import (square, rectangle, diamond, disk, cube, octahedron, ball, octagon, star)
###########################  HSV  ####################################

def segmentation (image):
    global minc,minr,width_,height_
    width=image.shape[0]
    height=image.shape[1]
    
    
    # conversion HSV
    img_hsv=rgb2hsv(image)
    #extracton h, s, v
    img_sat=img_hsv[:,:,1]
    
    #seuillage
    img_seuil=np.copy(image)
    for x in range(width): #Itération du X pixels
         for y in range(height): #Itération du Y pixels
            if img_sat[x][y]<0.4:
                img_seuil[x][y]=0
            
    #fermeture sur l'image seuillée
    # kernel_f = np.ones((5, 5), np.uint8)
    kernel_f=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img_seuil_f = cv2.morphologyEx(img_seuil, cv2.MORPH_CLOSE, kernel_f)
                
    #ouverture 
    # kernel_o = np.ones((7,7), np.uint8); 
    kernel_o=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    img_seuil_f_o = cv2.morphologyEx(img_seuil_f, cv2.MORPH_OPEN, kernel_o)
    
    ####################################
    
    img_seuil_g=rgb2gray(img_seuil_f)
    
    thresh = threshold_otsu(img_seuil_g)
    bw = closing(img_seuil_g > thresh, square(3))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    
    # label image regions
    label_image = label(cleared)
    
    
    n=0
    for region in regionprops(label_image):
    # take regions with large enough areas
        if region.area >= 300 and n==0:
            n=1
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            width_=maxc - minc
            height_=maxr - minr
            
    
    ###############################################
            
    left = minc
    top = minr
    width_c = width_
    height_c = height_
    cadre=(left, top, left+width_c, top+height_c)
    
    #im1 = image.crop(cadre)
    im1 = image[left:left+width_c, top:top+height_c]
    
    
    im1_hsv=rgb2hsv(im1)
    im1_sat=im1_hsv[:,:,1]
    
    width1=im1.shape[0]
    height1=im1.shape[1]
    
    
    im1_seuil=np.copy(im1)
    for x in range(width1): #Itération du X pixels
         for y in range(height1): #Itération du Y pixels
            if im1_sat[x][y]<0.4:
                im1_seuil[x][y]=0
                #im1_teinte[y][x]=0
            
                     
    #fermeture sur l'image seuillée
    # kernel_f = np.ones((3, 3), np.uint8)
    kernel_f=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    # kernel_f=disk(1)
    im1_seuil_f = cv2.morphologyEx(im1_seuil, cv2.MORPH_CLOSE, kernel_f)
    
    return im1_seuil_f 

