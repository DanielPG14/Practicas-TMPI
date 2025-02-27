import cv2
import numpy as np
import imutils
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)
plt.ion()#Activa el modo interactivo
fig,ax=plt.subplots(2,2)#Se inicializa la grafica
bg = None

# COLORES PARA VISUALIZACIÓN
color_start = (204,204,0)
color_end = (204,0,204)
color_far = (255,0,0)

color_start_far = (204,204,0)
color_far_end = (204,0,204)
color_start_end = (0,255,255)

color_contorno = (0,255,0)
color_ymin = (0,130,255) # Punto más alto del contorno
color_fingers = (0,255,255)

# Definir rango de color rojo en espacio HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

while True:
    # Redimensionar la imagen para que tenga un ancho de 640
    ref, frame = video.read()
    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()
    if bg is not None:
        # Determinar la región de interés
        ROI = frame[50:300, 380:600]
        cv2.rectangle(frame, (380-2, 50-2), (600+2, 300+2), color_fingers, 1)
        grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

        # Región de interés del fondo de la imagen
        bgROI = bg[50:300, 380:600]

        # Determinar la imagen binaria (background vs foreground)
        dif = cv2.absdiff(grayROI, bgROI)
        _, th = cv2.threshold(dif, 30, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th, 7)

        # Encontrando los contornos de la imagen binaria
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

        for cnt in cnts:
            # Encontrar el centro del contorno
            M = cv2.moments(cnt)
            if M["m00"] == 0: M["m00"] = 1
            x = int(M["m10"]/M["m00"])
            y = int(M["m01"]/M["m00"])
            cv2.circle(ROI, (x, y), 5, (0,255,0), -1)

            # Punto más alto del contorno
            ymin = cnt.min(axis=1)
            cv2.circle(ROI, tuple(ymin[0]), 5, color_ymin, -1)

            # Contorno encontrado a través de cv2.convexHull
            hull1 = cv2.convexHull(cnt)
            cv2.drawContours(ROI, [hull1], 0, color_contorno, 2)

            # Defectos convexos
            hull2 = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull2)

            if defects is not None:
                inicio = [] # Contenedor de puntos iniciales
                fin = [] # Contenedor de puntos finales

                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = cnt[s][0]
                    end = cnt[e][0]
                    far = cnt[f][0]

                    # Calcular el ángulo asociado al defecto convexo
                    a = np.linalg.norm(far - end)
                    b = np.linalg.norm(far - start)
                    c = np.linalg.norm(start - end)

                    angulo = np.arccos((np.power(a, 2) + np.power(b, 2) - np.power(c, 2)) / (2 * a * b))
                    angulo = np.degrees(angulo)
                    angulo = int(angulo)

                    # Condición para descartar defectos convexos no deseados
                    if np.linalg.norm(start - end) > 20 and angulo < 90 and d > 12000:
                        inicio.append(start)
                        fin.append(end)

                        # Visualización de los puntos de inicio y final
                        cv2.circle(ROI, tuple(start), 5, color_start, 2)
                        cv2.circle(ROI, tuple(end), 5, color_end, 2)
                        cv2.circle(ROI, tuple(far), 7, color_far, -1)

        # Detección de color rojo en la misma región de interés (ROI)
        hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)  # Convertir a espacio de color HSV
        mask_red = cv2.inRange(hsv, lower_red, upper_red)  # Crear una máscara para el color rojo
        red_res = cv2.bitwise_and(ROI, ROI, mask=mask_red)  # Aplicar la máscara a la ROI
        red_cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Histograma
        canales_de_color=cv2.split(frame)#Se divide el frame en canales de color
        canales_de_colorROI=cv2.split(ROI)#Se divide el ROI en canales de color
        #Subgrafica para ver el video completo
        ax[0,0].clear()#Limpiar la subgrafica
        ax[0,0].imshow(frame)
        ax[0,0].axis('off')
        ax[0,0].set_title("Video completo")
        #Subgrafica para ver el histograma del video completo
        ax[0,1].clear()
        ax[0,1].set_xlim([0,256])
        ax[0,1].set_title("Histograma del video completo")
        ax[0,1].set_xlabel("Intensidad de pixel")
        ax[0,1].set_ylabel("Frecuencia")
        colores=("b","g","r")
        for (canal,color) in zip (canales_de_color,colores): #Zip es para recorrer dos listas al mismo tiempo
            hist=cv2.calcHist([canal],[0], None, [256], [0,256])
            ax[0,1].plot(hist,color=color)
        #Subgrafica para ver sección ROI
        ax[1,0].clear()#Limpiar la subgrafica
        ax[1,0].imshow(ROI)
        ax[1,0].axis('off')
        ax[1,0].set_title("Video mano")
        #Subgrafica para ver el histograma de la sección ROI
        ax[1,1].clear()
        ax[1,1].set_xlim([0,256])
        ax[1,1].set_title("Histograma de la sección de la mano")
        ax[1,1].set_xlabel("Intensidad de pixel")
        ax[1,1].set_ylabel("Frecuencia")
        for (canal,color) in zip (canales_de_colorROI,colores): #Zip es para recorrer dos listas al mismo tiempo
            hist=cv2.calcHist([canal],[0], None, [256], [0,256])
            ax[1,1].plot(hist,color=color)
        plt.pause(0.001)

        # Dibujar contornos rojos detectados
        for red_cnt in red_cnts:
            if cv2.contourArea(red_cnt) > 500:  # Filtrar contornos pequeños
                cv2.drawContours(ROI, [red_cnt], -1, (0, 0, 255), 2)  # Dibujar el contorno rojo

        cv2.imshow('Red Color Detection', red_res)
        cv2.imshow('Threshold', th)

    cv2.imshow('Frame', frame)

    k = cv2.waitKey(20)
    print("Se presiono{k}")

    if k == ord('i') and bg is None:  # Solo inicializar el fondo una vez
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
plt.ioff()#Desactiva modo interactivo
plt.show()#Muestra la ultima grafica