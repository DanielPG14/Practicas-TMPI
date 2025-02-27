import cv2
import numpy as np
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)
ret,frame=video.read()
imagen_gris=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
grisB=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

if video is None:
    print("Sin video")
else:
    while True:
        #Método para calculo de histograma
        hist=cv2.calcHist([grisB],[0], None, [256], [0,256])
        #Método para limpiar el grafico
        plt.clf()
        #Metodo para hacer la grafica
        plt.plot(hist)
        plt.xlim([0,256])
        plt.title("Histograma")
        plt.xlabel("Intensidad de pixel")
        plt.ylabel("Frecuencia")
        #plt.hist(frame.ravel(),256,[0,256])
        #Metodo para mostrar la grafica
        plt.pause(0.001)
        cv2.imshow("video",frame)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break

video.release()
cv2.destroyAllWindows()
