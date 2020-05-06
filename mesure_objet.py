# -*- coding: utf-8 -*-
"""
Created on Tue May  5 12:26:18 2020

@author: Win7
"""

def mesure_objet(image):    

    return min(image.shape[0],image.shape[1])/max(image.shape[0],image.shape[1])
