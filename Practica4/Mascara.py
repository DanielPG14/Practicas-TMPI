import cv2
import numpy as np

imagen=cv2.imread("Practica4/personas.jpg")

def mostrar_imagen_y_mascara(imagen, mascara,num):
    mascara=mascara.astype(np.uint8)
    mascara_color=cv2.cvtColor(mascara,cv2.COLOR_GRAY2BGR)
    mascara_color[:,:,1:3]=0
    cv2.imshow(f'Imagen Original - {num}', imagen)
    cv2.imshow(f'Imagen Mascara - {num}', mascara_color)

#Mascara color
x,y,w,h=100,100,200,150
mascara_rectangular=np.zeros(imagen.shape[:2],dtype=np.uint8)
cv2.rectangle(mascara_rectangular,(x,y),(x+w, y+h),255,-1)
mostrar_imagen_y_mascara(imagen,mascara_rectangular,1)

#Mascara circular
centro=(200,200)
radio=100
mascara_circular=np.zeros(imagen.shape[:2],dtype=np.uint8)
cv2.circle(mascara_circular,centro,radio,255,-1)
mostrar_imagen_y_mascara(imagen,mascara_circular,2)

#Mascara eliptica
ejex, ejey = 150, 100
mascara_eliptica = np.zeros(imagen.shape[:2], dtype=np.uint8)
cv2.ellipse(mascara_eliptica, (200, 200), (ejex, ejey), 0, 0, 360, 255, -1)
mostrar_imagen_y_mascara(imagen, mascara_eliptica, 3)

#Mascara aleatoria
mascara_aleatoria=np.random.randint(0,2,size=imagen.shape[:2],dtype=np.uint8)*255
mostrar_imagen_y_mascara(imagen,mascara_aleatoria,4)

#Mascara borde Canny
imagen_gris=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
bordes_canny = cv2.Canny(imagen_gris,100,200)
mostrar_imagen_y_mascara(imagen,bordes_canny,5)

#Mascarta borde Sobel
sobelx=cv2.Sobel(imagen_gris,cv2.CV_64F,1,0,ksize=5)
sobely=cv2.Sobel(imagen_gris,cv2.CV_64F,0,1,ksize=5)
gradiente=np.sqrt(sobelx**2+sobely**2)
_,mascara_sobel=cv2.threshold(gradiente,50,55,cv2.THRESH_BINARY)
mostrar_imagen_y_mascara(imagen,mascara_sobel,6)

#Mascara umbral adaptativo
mascara_umbral_adaptativo=cv2.adaptiveThreshold(imagen_gris,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
mostrar_imagen_y_mascara(imagen,mascara_umbral_adaptativo,7)

#Mascara euclidiana
dist_transform=cv2.distanceTransform(imagen_gris,cv2.DIST_L2,5)
_,mascara_euclidiana=cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
mostrar_imagen_y_mascara(imagen,mascara_euclidiana,8)

#Mascarta borde Sobel modificada
sobelx=cv2.Sobel(imagen_gris,cv2.CV_64F,1,0,ksize=3)
sobely=cv2.Sobel(imagen_gris,cv2.CV_64F,0,1,ksize=3)
gradiente=np.sqrt(sobelx**2+sobely**2)
_,mascara_sobel=cv2.threshold(gradiente,25,100,cv2.THRESH_BINARY)
mostrar_imagen_y_mascara(imagen,mascara_sobel,9)

key=cv2.waitKey(0)
if key==ord('q'):
    cv2.destroyAllWindows()
