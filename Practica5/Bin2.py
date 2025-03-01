import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    frame = imutils.resize(frame, width=640) # Redimensionar el video
    if ret==False: break
    img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Conversion a escala de grises
    img=cv2.medianBlur(img,5) #Filtro de mediana
    _,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #Umbralización global 
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) #Umbralización adaptativa
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #Umbralización adaptativa
    cv2.imshow("Original Image", img)
    cv2.imshow("Thresholding 127", th1)
    cv2.imshow("Thresholding Adaptativo", th2)
    cv2.imshow("Thresholding Gaussiano Adaptativo", th3)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
video.release()
cv2.destroyAllWindows()