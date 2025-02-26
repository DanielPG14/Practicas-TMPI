import cv2
import numpy as np

video = cv2.VideoCapture(0)

# Definir los rangos de colores en HSV
yellow_bajo = np.array([15, 100, 100], np.uint8)
yellow_alto = np.array([45, 255, 255], np.uint8)

green_bajo = np.array([46, 100, 100], np.uint8)
green_alto = np.array([90, 225, 255], np.uint8)

blue_bajo = np.array([91, 100, 100], np.uint8)
blue_alto = np.array([130, 225, 255], np.uint8)

red_bajo1 = np.array([0, 100, 100], np.uint8)
red_alto1 = np.array([10, 225, 255], np.uint8)

red_bajo2 = np.array([175, 100, 100], np.uint8)
red_alto2 = np.array([180, 225, 255], np.uint8)

if video is None:
    print("No se encontro imagen")
else:
    while True:
        ret,frame=video.read()
        #Mascara borde Canny
        imagen_HSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        imagen_gris=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Detectar color rojo (dos rangos)
        mask_red1 = cv2.inRange(imagen_HSV, red_bajo1, red_alto1)
        mask_red2 = cv2.inRange(imagen_HSV, red_bajo2, red_alto2)
        mask_red = cv2.add(mask_red1, mask_red2) 
        mask_redbis = cv2.bitwise_and(frame, frame, mask=mask_red)

        # Detectar color azul
        mask_blue = cv2.inRange(imagen_HSV, blue_bajo, blue_alto)
        mask_bluebis = cv2.bitwise_and(frame, frame, mask=mask_blue)

        # Detectar color verde
        mask_green = cv2.inRange(imagen_HSV, green_bajo, green_alto)
        mask_greenbis = cv2.bitwise_and(frame, frame, mask=mask_green)

        # Detectar color amarillo
        mask_yellow = cv2.inRange(imagen_HSV, yellow_bajo, yellow_alto)
        mask_yellowbis = cv2.bitwise_and(frame, frame, mask=mask_yellow)

        #Bordes Canny y contornos rojos
        imagen_gris_red = cv2.cvtColor(mask_redbis, cv2.COLOR_BGR2GRAY)
        gaussiana = cv2.GaussianBlur(imagen_gris_red, (5,5), 0) #Aplicar filtro gaussiano para suavizar la imagen
        bordes_cannyR = cv2.Canny(gaussiana,50,250)
        (contornosR,_) = cv2.findContours(bordes_cannyR.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contornosR, -1, (0,0,255), 2)

        #Bordes Canny y contornos azul
        imagen_gris_blue = cv2.cvtColor(mask_bluebis, cv2.COLOR_BGR2GRAY) 
        gaussiana = cv2.GaussianBlur(imagen_gris_blue, (5,5), 0) #Aplicar filtro gaussiano para suavizar la imagen
        bordes_cannyA = cv2.Canny(gaussiana,50,250)
        (contornosA,_) = cv2.findContours(bordes_cannyA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contornosA, -1, (255,0,0), 2)

        #Bordes Canny y contornos verde
        imagen_gris_green = cv2.cvtColor(mask_greenbis, cv2.COLOR_BGR2GRAY)
        gaussiana = cv2.GaussianBlur(imagen_gris_green, (5,5), 0)
        bordes_cannyV = cv2.Canny(gaussiana,50,250)
        (contornosV,_) = cv2.findContours(bordes_cannyV.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contornosV, -1, (0,255,0), 2)

        cv2.imshow('Imagen original', frame)
        cv2.imshow('Rojo', mask_red1)
        cv2.imshow('Azul', mask_blue)
        cv2.imshow('Verde', mask_green)

        key=cv2.waitKey(1)
        if key==ord('q'):
            break

video.release()            
cv2.destroyAllWindows()