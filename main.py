import cv2 as cv
from PIL import Image as pil
from segmentation import segmentation
from texture import texture
from mesure_objet import mesure_objet
from Couleur import describe
from classfication import test,train
from results import results
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from os import listdir,chdir

randomforest = RandomForestClassifier()

randomforest = train(randomforest)

#test(randomforest)

## Matrice de confusion ##
chdir('images/testing')
imgs = listdir()

x_test = []

for file in imgs:
    features = []
    img = pil.open(file)
    imgseg = segmentation(img)
    tex = [i for i in texture(imgseg).tolist()]
    color = [i for i in describe(imgseg,8).tolist()]
    shape =  mesure_objet(imgseg)
    features = tex
    features.append(shape)
    features += color
    x_test.append(features)

X=np.array(x_test)

Y=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3])

results(Y,randomforest.predict(X))



