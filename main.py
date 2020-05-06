import cv2 as cv
from segmentation import segmentation
from texture import texture
from mesure_objet import mesure_objet
from classfication import classify
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from os import listdir,chdir
dic = {"Apple": 0, "Banane" : 1, "Orange" : 2, "Tomate" : 3}
reps = listdir("res2")
x=[]
y=[]
for fruit in reps:
    chdir('res2/'+fruit)
    imgs = listdir()
    for image in imgs:
        print(image)
        img = cv.imread(image)
        imgseg = segmentation(img)
        tex = texture(imgseg)
        shape =  mesure_objet(imgseg)
        tex2 = [i for i in tex.tolist()]
        tex2.append(shape)
        x.append(tex2)
        y.append(dic[fruit])
    chdir('../..')
X=np.array(x)
Y=np.array(y)
randomforest=RandomForestClassifier(n_estimators=30)
score = classify(X,Y,randomforest)

pommeTest = cv.imread("res/Apple/Apple 122.png")
bananeTest = cv.imread("res/Banane/Banana094.png")
orangeTest = cv.imread("res/Orange/Orange0094.png")
tomateTest = cv.imread("res/Tomate/Tamotoes0072.png")

imgseg = segmentation(tomateTest)
tex = texture(imgseg)
shape =  mesure_objet(imgseg)
tex2 = [i for i in tex.tolist()]
tex2.append(shape)
X=np.array(tex2).reshape(1, -1)
print(randomforest.predict(X))
