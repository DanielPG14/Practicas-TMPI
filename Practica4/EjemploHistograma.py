import cv2
import numpy as np
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)
ret,frame=video.read()
imagen_gris=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
grisB=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#Método para calculo de histograma
hist=cv2.calcHist([grisB],[0], None, [256], [0,256])

#Función para hacer la grafica del histograma
#plt.plot(hist)
plt.hist(frame.ravel(),256,[0,256])
#Función para mostrar la grafica
plt.show()
