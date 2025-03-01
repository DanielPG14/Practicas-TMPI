import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread("personas.jpg", cv2.IMREAD_GRAYSCALE) #Conversion a escala de grises
img=cv2.medianBlur(img,5) #Filtro de mediana
_,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY) #Umbralización global 
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2) #Umbralización adaptativa
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #Umbralización adaptativa
titles = ['Original Image', 'Thresholding 127',
            'Thresholding Adaptativo', 'Thresholding Gaussiano Adaptativo']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
