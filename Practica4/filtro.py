import cv2
import numpy as np

imagen=cv2.imread("Practica4/personas.jpg")

if imagen is None:
    print("No se encontro imagen")
else:
    #Filtro suavizado
    blur_image=cv2.blur(imagen,(5,5))

    cv2.imshow('Imagen Original', imagen)
    cv2.imshow('Imagen blur', blur_image)

    #Filtro negativo
    row,col,_=imagen.shape
    negativo=np.zeros((row,col,3),dtype=np.uint8)
    for a in range(0,row):
        for b in range(0,col):
            negativo[a,b,:]= 255-imagen[a,b,:]
    cv2.imshow('Imagen negativa', negativo)

    #Filtro pencilSketch
    gris, color=cv2.pencilSketch(imagen,sigma_s=60,sigma_r=0.7,shade_factor=0.05)
    cv2.imshow('Imagen gris', gris)
    cv2.imshow('Imagen color',color)

    #Filtro sepia
    #copia=imagen.copy()
    #copia=cv2.transform(copia,np.matrix([0.272,0.534,0.131],[0.349,0.286,0.168],[0.393,0.769,0.189]))
    #copia[np.where(copia>255)]=255
    #copia=np.array(copia,dtype=np.uint8)
    #cv2.imshow('Imagen sepia',copia)
    copia = imagen.copy().astype(np.float32)  # Convertir la imagen a float para cálculos
    sepia_matrix = np.array([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168],[0.393, 0.769, 0.189]], dtype=np.float32)# Matriz de transformación sepia
    copia = cv2.transform(copia, sepia_matrix)# Aplicar transformación
    copia = np.clip(copia, 0, 255)# Asegurar que los valores estén en el rango correcto
    copia = copia.astype(np.uint8)# Convertir de nuevo a uint8
    cv2.imshow('Imagen sepia', copia)

    #Filtro cartoon
    borde=cv2.bitwise_not(cv2.Canny(imagen,100,200))
    grisB=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
    grisB=cv2.medianBlur(grisB,5)
    borde2=cv2.adaptiveThreshold(grisB,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,7)
    dst=cv2.edgePreservingFilter(imagen,flags=2,sigma_s=64,sigma_r=0.25)
    cartoon1=cv2.bitwise_and(dst,dst,mask=borde)
    cartoon2=cv2.bitwise_and(dst,dst,mask=borde2)
    cv2.imshow('Imagen cartoon 1', cartoon1)
    cv2.imshow('Imagen cartoon 2', cartoon2)
    
    #Filtro con mascara sobel
    sobelx=cv2.Sobel(gris,cv2.CV_64F,1,0,ksize=3)
    sobely=cv2.Sobel(gris,cv2.CV_64F,0,1,ksize=3)
    gradiente=np.sqrt(sobelx**2+sobely**2)
    cv2.imshow('Imagen borde sobel', gradiente)

    key=cv2.waitKey(0)
    if key==ord('q'):
        cv2.destroyAllWindows()