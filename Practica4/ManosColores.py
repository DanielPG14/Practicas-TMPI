import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)
plt.ion()#Activa el modo interactivo
fig,ax=plt.subplots(2,2)#Se inicializa la grafica

# Definir los rangos de colores en HSV
Amar_bajo = np.array([15, 100, 100], np.uint8)
Amar_alto = np.array([45, 255, 255], np.uint8)

Verde_bajo = np.array([46, 100, 100], np.uint8)
Verde_alto = np.array([90, 225, 255], np.uint8)

Azul_bajo = np.array([91, 100, 100], np.uint8)
Azul_alto = np.array([130, 225, 255], np.uint8)

rojo_bajo1 = np.array([0, 100, 100], np.uint8)
rojo_alto1 = np.array([10, 225, 255], np.uint8)

rojo_bajo2 = np.array([175, 100, 100], np.uint8)
rojo_alto2 = np.array([180, 225, 255], np.uint8)

if video is None:
    print("No se encontro imagen")
else:
    while True:
        ret,frame=video.read()
        frame = imutils.resize(frame, width=640, height=580) # Redimensionar el video
        frame = cv2.flip(frame, 1) # Espejo de la imagen
        frameAux = frame.copy()
        CuadroI = frame[50:300, 380:550] # Región de interés
        cv2.rectangle(frame, (380, 50), (550, 300), (170,95,92),1)#Recibe imagen, coordenadas, color y grosor
        GrisCuadroI=cv2.cvtColor(CuadroI,cv2.COLOR_BGR2GRAY)#Convertir a escala de grises

        FCI=cv2.cvtColor(frameAux,cv2.COLOR_BGR2GRAY)
        Fondo_CuadroI=FCI[50:300, 380:550]#Recorte de la imagen de fondo
        Difer=cv2.absdiff(GrisCuadroI,Fondo_CuadroI)#Diferencia entre la imagen de fondo y la actual
        _, masc_sob = cv2.threshold(Difer, 30, 255, cv2.THRESH_BINARY) #Umbralización mediante mascara sobel
        masc_sob = cv2.medianBlur(masc_sob, 7) #Aplicar filtro de suavizado

        cv2.imshow("Video Original",frame)
        cv2.imshow("Mano",GrisCuadroI)
        cv2.imshow("FC",Fondo_CuadroI)
        cv2.imshow("Sustracción del fondo",Difer)
        cv2.imshow("Umbralización",masc_sob)

        #Mascara borde Canny
        imagen_HSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        imagen_gris=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Detectar color rojo (dos rangos)
        mask_red1 = cv2.inRange(imagen_HSV, rojo_bajo1, rojo_alto1)
        mask_red2 = cv2.inRange(imagen_HSV, rojo_bajo2, rojo_alto2)
        mask_red = cv2.add(mask_red1, mask_red2) 
        mask_redbis = cv2.bitwise_and(frame, frame, mask=mask_red)

        # Detectar color azul
        mask_blue = cv2.inRange(imagen_HSV, Azul_bajo, Azul_alto)
        mask_bluebis = cv2.bitwise_and(frame, frame, mask=mask_blue)

        # Detectar color verde
        mask_green = cv2.inRange(imagen_HSV, Verde_bajo, Verde_alto)
        mask_greenbis = cv2.bitwise_and(frame, frame, mask=mask_green)

        # Detectar color amarillo
        mask_yellow = cv2.inRange(imagen_HSV, Amar_bajo, Amar_alto)
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

        key=cv2.waitKey(1)
        if key==ord('q'):
            break

video.release()            
cv2.destroyAllWindows()