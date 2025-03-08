import cv2
import numpy as np

# Datos 0 gatos 1 perros
datos = np.array([[150,120,130], [200,100,170], [100,80,90], [250,240,230], [160,140,150], [90,70,80]], dtype=np.float32)
etiquetas = np.array([[0],[1],[0],[1],[0],[0]], dtype=np.float32)

# Crear la red neuronal
rna=cv2.ml.ANN_MLP_create()
rna.setLayerSizes(np.array([3, 5, 1], dtype=np.int32))
rna.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM,1,1)
rna.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP,0.1,0.1)
rna.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10000, 0.01))
rna.train(datos, cv2.ml.ROW_SAMPLE, etiquetas)

img = cv2.imread(r'c:\Users\PC\OneDrive\Documentos\Códigos\Octavo\PDI\Practicas-TMPI\Practica7\perroygato2.jpg')
img_copia = img.copy()

def seleccionar_roi(event,x,y,flags,param):
    global ROI
    if event == cv2.EVENT_LBUTTONDOWN:
        ROI = cv2.selectROI('imagen original',img_copia,False,False)
        cv2.destroyWindow('imagen original')

cv2.imshow('imagen original',img_copia)
cv2.setMouseCallback('imagen original',seleccionar_roi)
cv2.waitKey(0)

x,y,w,h = ROI
region_selecionada = img[y:y+h,x:x+w]
pixeles = region_selecionada.reshape(-1,3).astype(np.float32)

promedio_color=np.mean(pixeles, axis=0).reshape(1,-1).astype(np.float32)

_, resultado = rna.predict(promedio_color)
clasificacion = 'Gato' if resultado[0][0] > 0.5 else 'Perro'

cv2.putText(img,clasificacion,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
cv2.imshow('clasificación',img)
cv2.waitKey(0)
cv2.destroyAllWindows()