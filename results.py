from os import listdir
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas
from decimal import Decimal


# Affiche la matrice de confusion des resultats avec la précision de l'algo
# (Optionel) Affiche les images avec la prédiction si un dossier d'images est spécifié

def results(y_true,y_pred,dir_im = ""):
	dic = { 0 : "Pomme", 1 : "Banane", 2 : "Orange" , 3 : "Tomate" }
	fruits = ["Pomme","Banane","Orange","Tomate"]

	# Matrice de confusion
	conf = confusion_matrix(y_true,y_pred)
	C = pandas.DataFrame(conf,index=fruits,columns=fruits )
	print(" ------- Matrice de confusion pour la base de test ------- ")
	print("\n",C)
	
	# Precision de l'algorithme
	p = 0
	m = len(y_true) # nombre d'images 
	for n in range(0,m):
		if y_true[n] == y_pred[n]:
			p += 1
	p /= m	
	p = Decimal(p)
	p = round(p,2)
	print("\nPrecision de l'algorithme : ",100*p,"%\n")

	# Affichage des images avec prédictions
	if dir_im != "":
		n_im = listdir(dir_im)
		j=0
		for i in n_im:
			image = cv.imread("images/testing/"+i)
			cv.putText(image, dic[y_pred[j]], (10, 30), cv.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)
			cv.imshow("Image", image)
			cv.waitKey(0)
			j += 1


# Code test (Appuyer sur espace pour faire defiler les images)
#-------------------------------------
# y_true = [0,0,0,0,0,0,1,1,2,2,3,3]
# y_pred = [0,0,0,2,0,1,2,1,2,2,3,0]
# dir_im = "images/testing/"
# results(y_true,y_pred,dir_im) # Avec images
# #results(y_true,y_pred) # Sans images





