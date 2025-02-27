import cv2
import numpy as np
from matplotlib import pyplot as plt

imagen=cv2.imread("Practica4/personas.jpg")
imagen_gris=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)

#Método para calculo de histograma
hist=cv2.calcHist([imagen_gris],[0], None, [256], [0,256])

#Función para hacer la grafica del histograma
#plt.plot(hist)
plt.hist(imagen.ravel(),256,[0,256])
#Función para mostrar la grafica
plt.show()
