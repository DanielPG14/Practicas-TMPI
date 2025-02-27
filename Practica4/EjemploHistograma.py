import cv2
import numpy as np
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)
plt.ion()#Activa el modo interactivo
fig,ax=plt.subplots()#Se inicializa la grafica

if video is None:
    print("Sin video")
else:
    while True:
        ret,frame=video.read()
        grisB=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #Método para calculo de histograma
        hist=cv2.calcHist([grisB],[0], None, [256], [0,256])
        #limpiar el grafico
        ax.clear()
        #Metodo para hacer la grafica
        ax.plot(hist)
        ax.set_xlim([0,256])
        ax.set_title("Histograma")
        ax.set_xlabel("Intensidad de pixel")
        ax.set_ylabel("Frecuencia")
        #plt.hist(frame.ravel(),256,[0,256])
        #Metodo para mostrar la grafica
        plt.pause(0.001)
        cv2.imshow("video",grisB)
        key=cv2.waitKey(1)
        if key==ord('q'):
            break

video.release()
cv2.destroyAllWindows()
plt.ioff()#Desactiva el modo interactivo
plt.show()#Ultia grafica