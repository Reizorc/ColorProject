# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:37:28 2020

@author: Win7
"""
import tkinter as tk
from PIL import Image as pil 
from PIL import ImageTk as pol
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

def mesure_couleur(image):
    im1_hsv=rgb2hsv(image)
    im1_teinte=im1_hsv[:,:,0]
    im1_sat=im1_hsv[:,:,1]
    
    width1=image.shape[1]
    height1=image.shape[0]
    
    n=0
    compteur_couleur=0
    for x in range(0,width1): #Itération du X pixels
         for y in range(0,height1): #Itération du Y pixels
            if im1_sat[y][x]>=0.4:
                compteur_couleur=compteur_couleur+im1_teinte[y][x]
                n=n+1
    couleur=compteur_couleur/n  
    if couleur<0.08:
        return ('rouge')
    elif couleur<0.125:
        return ('orange')
    elif couleur<0.16:
        return ('marron')
    elif couleur<0.20:
        return ('jaune')
    elif couleur<0.41:
        return ('vert')    
    elif couleur<0.75:
        return ('bleu')
    else:
        return ('rouge')