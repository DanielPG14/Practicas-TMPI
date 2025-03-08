import cv2
import numpy as np

# Datos de entrenamiento
datos = np.array([[100,150], [200,250], [300,350], [400,450], [500,550]], dtype=np.float32)
etiquetas = np.array([[0],[1],[0],[1],[0]], dtype=np.float32)

# Crear la red neuronal
rna=cv2.ml.ANN_MLP_create()
rna.setLayerSizes(np.array([2, 5, 1], dtype=np.int32))
rna.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
rna.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
rna.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 10000, 0.01))
rna.train(datos, cv2.ml.ROW_SAMPLE, etiquetas)

# Nueva muestra para predicción
nueva_muestra = np.array([[350,400]], dtype=np.float32)
_, resultado = rna.predict(nueva_muestra)

print("Resultado de la predicción:", resultado)

img=cv2.imread('Practica7/gato.jpg')
gris=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bordes=cv2.Canny(gris, 100, 200)
puntos=np.column_stack(np.where(bordes>0)).astype(np.float32)

_, clasificacion = rna.predict(puntos)

# Dibujar puntos clasificados en la imagen
for (x, y), clas in zip(puntos, clasificacion):
    color = (0, 255, 0) if clas > 0 else (0, 0, 255)
    cv2.circle(img, (x, y), 5, color, -1)

cv2.imshow('Clasificación RNA', img)
cv2.waitKey(0)