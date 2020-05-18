import cv2 as cv
from segmentation import segmentation
from texture import texture
from mesure_objet import mesure_objet
from Couleur import describe
from os import listdir,chdir
import numpy as np
from PIL import Image as pil

def train(forest):
    dic = {"Apple": 0, "Banana" : 1, "Orange" : 2, "Tomato" : 3}
    x=[]
    y=[]
    chdir('images/training')
    reps = listdir()
    for fruit in reps:
        chdir(fruit)
        imgs = listdir()
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
            x.append(features)
            y.append(dic[fruit])
        chdir('..')
    X=np.array(x)
    Y=np.array(y)
    forest.fit(X,Y)
    print(" ------- Résultat pour la base d'entrainement  ------- ")
    print("Taux de réussite : " +str(forest.score(X,Y)/1*100)+"%")
    chdir('../..')
    return forest

def test(forest):
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
    print(" ------- Résultat pour la base de test ------- ")
    print(randomforest.predict(X))

    chdir('../..')
    return None
    
