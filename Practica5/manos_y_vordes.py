import cv2
import numpy as np
import imutils

video = cv2.VideoCapture(0)
bg = None

# COLORES PARA VISUALIZACIÓN
color_start = (204,204,0)
color_end = (204,0,204)
color_far = (255,0,0)

color_start_far = (204,204,0)
color_far_end = (204,0,204)
color_start_end = (0,255,255)

color_contorno = (0,255,0)
color_ymin = (0,130,255)  # Punto más alto del contorno
color_fingers = (0,255,255)

# Definir rango de color rojo en espacio HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

detect_hand = False
detect_colors = False

while True:
    ref, frame = video.read()
    frame = imutils.resize(frame, width=640)
    frame = cv2.flip(frame, 1)
    frameAux = frame.copy()

    # Si la detección de manos está activada
    if detect_hand and bg is not None:
        # Detección de manos
        # Determinar la región de interés
        ROI = frame[50:300, 380:600]
        cv2.rectangle(frame, (378, 48), (602, 302), (0, 255, 255), 2)
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
                inicio = []  # Contenedor de puntos iniciales
                fin = []  # Contenedor de puntos finales

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

        # Mostrar la imagen procesada sin la parte de detección del color rojo
        cv2.imshow('Threshold', th)

    # Si la detección de colores está activada
    if detect_colors:
        # Detección de bordes de colores
        ROI = frame[50:300, 380:600]
        cv2.rectangle(frame, (378, 48), (602, 302), (0, 255, 255), 2)

        # Convertir ROI a HSV
        hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)

        # Aplicar las máscaras para detectar los diferentes colores
        mask_red = cv2.inRange(hsv, lower_red, upper_red)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Encontrar los contornos de los colores
        red_cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_cnts, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_cnts, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibujar los contornos de cada color en el ROI
        for red_cnt in red_cnts:
            if cv2.contourArea(red_cnt) > 500:  # Filtrar contornos pequeños
                cv2.drawContours(ROI, [red_cnt], -1, (0, 0, 255), 2)  # Rojo en el contorno

        for yellow_cnt in yellow_cnts:
            if cv2.contourArea(yellow_cnt) > 500:  # Filtrar contornos pequeños
                cv2.drawContours(ROI, [yellow_cnt], -1, (0, 255, 255), 2)  # Amarillo en el contorno

        for blue_cnt in blue_cnts:
            if cv2.contourArea(blue_cnt) > 500:  # Filtrar contornos pequeños
                cv2.drawContours(ROI, [blue_cnt], -1, (255, 0, 0), 2)  # Azul en el contorno

        for green_cnt in green_cnts:
            if cv2.contourArea(green_cnt) > 500:  # Filtrar contornos pequeños
                cv2.drawContours(ROI, [green_cnt], -1, (0, 255, 0), 2)  # Verde en el contorno

    cv2.imshow('Frame', frame)  # Mostrar el frame con los contornos de todos los colores

    k = cv2.waitKey(20)
    if k == ord('i') and bg is None and not detect_hand:  # Solo inicializar el fondo una vez
        bg = cv2.cvtColor(frameAux, cv2.COLOR_BGR2GRAY)
        detect_hand = True  # Activar detección de manos
        cv2.imshow('Threshold', frame)  # Mostrar ventana extra para detección de manos
    elif k == ord('i') and detect_hand:  # Desactivar detección de manos
        detect_hand = False
        cv2.destroyWindow('Threshold')  # Cerrar la ventana extra de detección de manos
    elif k == ord('o'):  # Activar o desactivar la detección de colores
        detect_colors = not detect_colors
    elif k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
