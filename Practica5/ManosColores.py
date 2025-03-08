import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)
plt.ion()#Activa el modo interactivo
fig,ax=plt.subplots(2,2)#Se inicializa la grafica
FCI=None

# Definir los rangos de colores en HSV
Amar_bajo = np.array([20, 50, 50], np.uint8)
Amar_alto = np.array([40, 255, 255], np.uint8)

Verde_bajo = np.array([35, 50, 50], np.uint8)
Verde_alto = np.array([85, 255, 255], np.uint8)

Azul_bajo = np.array([90, 50, 50], np.uint8)
Azul_alto = np.array([130, 255, 255], np.uint8)

rojo_bajo1 = np.array([0, 50, 50], np.uint8)
rojo_alto1 = np.array([10, 255, 255], np.uint8)

rojo_bajo2 = np.array([170, 50, 50], np.uint8)
rojo_alto2 = np.array([180, 255, 255], np.uint8)

# Rango de color para la piel en HSV
piel_bajo = np.array([0, 20, 70], np.uint8)
piel_alto = np.array([20, 255, 255], np.uint8)

if video is None:
    print("No se encontro imagen")
else:
    while True:
        ret,frame=video.read()
        frame = imutils.resize(frame, width=640) # Redimensionar el video
        frame = cv2.flip(frame, 1) # Espejo de la imagen
        frameAux = frame.copy()

        if FCI is not None:
            CuadroI = frame[50:300, 380:550] # Región de interés
            cv2.rectangle(frame, (380, 50), (550, 300), (170,95,92),1)#Recibe imagen, coordenadas, color y grosor

            #Mascara borde Canny
            imagen_HSV=cv2.cvtColor(CuadroI,cv2.COLOR_BGR2HSV)#Convertir cuadro a HSV

            # Detectar color rojo (dos rangos)
            mascara_roja1 = cv2.inRange(imagen_HSV, rojo_bajo1, rojo_alto1)
            mascara_roja2 = cv2.inRange(imagen_HSV, rojo_bajo2, rojo_alto2)
            mascara_roja = cv2.add(mascara_roja1, mascara_roja2) 
            mascara_rojabis = cv2.bitwise_and(CuadroI, CuadroI, mask=mascara_roja)

            # Detectar color azul
            mascara_azul = cv2.inRange(imagen_HSV, Azul_bajo, Azul_alto)
            mascara_azulbis = cv2.bitwise_and(CuadroI, CuadroI, mask=mascara_azul)

            # Detectar color verde
            mascara_verde = cv2.inRange(imagen_HSV, Verde_bajo, Verde_alto)
            mascara_verdebis = cv2.bitwise_and(CuadroI, CuadroI, mask=mascara_verde)

            # Detectar color amarillo
            mascara_amar = cv2.inRange(imagen_HSV, Amar_bajo, Amar_alto)
            mascara_amarbis = cv2.bitwise_and(CuadroI, CuadroI, mask=mascara_amar)

            #Bordes Canny y contornos rojos
            imagen_gris_R = cv2.cvtColor(mascara_rojabis, cv2.COLOR_BGR2GRAY)
            gaussiana = cv2.GaussianBlur(imagen_gris_R, (5,5), 0) #Aplicar filtro gaussiano para suavizar la imagen
            bordes_cannyR = cv2.Canny(gaussiana,50,250)
            (contornosR,_) = cv2.findContours(bordes_cannyR.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contornosR:
                if cv2.contourArea(c) > 500:
                    cv2.drawContours(CuadroI, contornosR, -1, (0,0,255), 2)

            #Bordes Canny y contornos azul
            imagen_gris_A = cv2.cvtColor(mascara_azulbis, cv2.COLOR_BGR2GRAY) 
            gaussiana = cv2.GaussianBlur(imagen_gris_A, (5,5), 0) #Aplicar filtro gaussiano para suavizar la imagen
            bordes_cannyA = cv2.Canny(gaussiana,50,250)
            (contornosA,_) = cv2.findContours(bordes_cannyA.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contornosA:
                if cv2.contourArea(c) > 500: #Filtrar contornos pequeños
                    cv2.drawContours(CuadroI, contornosA, -1, (255,0,0), 2)

            #Bordes Canny y contornos verde
            imagen_gris_V = cv2.cvtColor(mascara_verdebis, cv2.COLOR_BGR2GRAY)
            gaussiana = cv2.GaussianBlur(imagen_gris_V, (5,5), 0)
            bordes_cannyV = cv2.Canny(gaussiana,50,250)
            (contornosV,_) = cv2.findContours(bordes_cannyV.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contornosV:
                if cv2.contourArea(c) > 500:
                    cv2.drawContours(CuadroI, contornosV, -1, (0,255,0), 2)

            #Bordes Canny y contornos amarillo
            imagen_gris_Y = cv2.cvtColor(mascara_amarbis, cv2.COLOR_BGR2GRAY)
            gaussiana = cv2.GaussianBlur(imagen_gris_Y, (5,5), 0)
            bordes_cannyY = cv2.Canny(gaussiana,50,250)
            (contornosY,_) = cv2.findContours(bordes_cannyY.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contornosY:
                if cv2.contourArea(c) > 500:
                    cv2.drawContours(CuadroI, contornosY, -1, (0,255,255), 2)

            #Histograma
            canales_de_color=cv2.split(frame)#Se divide el frame en canales de color
            canales_de_color_cuadro=cv2.split(CuadroI)#Se divide el ROI en canales de color
            #Subgrafica para ver el video completo
            ax[0,0].clear()#Limpiar la subgrafica
            ax[0,0].imshow(frame)
            ax[0,0].axis('off')
            ax[0,0].set_title("Video completo")
            #Subgrafica para ver el histograma del video completo
            ax[0,1].clear()
            ax[0,1].set_xlim([0,256])
            ax[0,1].set_title("Histograma del video completo")
            colores=("b","g","r")
            for (canal,color) in zip (canales_de_color,colores): #Zip es para recorrer dos listas al mismo tiempo
                hist=cv2.calcHist([canal],[0], None, [256], [0,256])
                ax[0,1].plot(hist,color=color)
            #Subgrafica para ver sección de interes
            ax[1,0].clear()#Limpiar la subgrafica
            ax[1,0].imshow(CuadroI)
            ax[1,0].axis('off')
            ax[1,0].set_title("Video mano")
            #Subgrafica para ver el histograma de la sección de interes
            ax[1,1].clear()
            ax[1,1].set_xlim([0,256])
            ax[1,1].set_title("Histograma de la sección de la mano")
            for (canal,color) in zip (canales_de_color_cuadro,colores): #Zip es para recorrer dos listas al mismo tiempo
                hist=cv2.calcHist([canal],[0], None, [256], [0,256])
                ax[1,1].plot(hist,color=color)
            plt.pause(0.001)
        
        cv2.imshow("Video Original",frame)

        key=cv2.waitKey(30)
        if key==ord('q'):
            print("Se presiono{k}")
            break
        if key==ord('w'):
            FCI=cv2.cvtColor(frameAux,cv2.COLOR_BGR2GRAY)

video.release()            
cv2.destroyAllWindows()
plt.ioff()#Desactiva el modo interactivo