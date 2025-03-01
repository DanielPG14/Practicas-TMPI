import cv2
import numpy as np
from matplotlib import pyplot as plt

cont=0
img=cv2.imread("Monedas.jpg")
cv2.imshow("Imagen original", img)
gris=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Convertir a escala de grises
gauss=cv2.GaussianBlur(gris,(5,5),0) #Filtro gaussiano
cv2.imshow("Imagen con filtro gaussiano", gauss)
canny=cv2.Canny(img, 50, 150) #Deteccion de bordes
cv2.imshow("Imagen con deteccion de bordes", canny)
contorno,_=cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Deteccion de contornos
for c in contorno:
    area=cv2.contourArea(c) #Calcular area de los contornos
    if area>200: #Filtrar contornos por area
        cv2.drawContours(img, [c], 0, (0,255,0), 2) #Dibujar contornos
        cont=cont+1
cv2.putText(img, f"Monedas encontradas: {cont}", (10,20), 1, 1, (0,0,255), 1) #Texto con el numero de monedas encontradas
#cv2.drawContours(img, contorno, -1, (0,255,0), 2) #Dibujar contornos
cv2.imshow("Imagen con contornos", img)
cv2.waitKey(0) 
cv2.destroyAllWindows()