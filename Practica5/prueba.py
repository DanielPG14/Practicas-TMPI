import cv2

# Probar diferentes índices (0, 1, 2, 3...)
for i in range(5):
    video = cv2.VideoCapture(i)
    if video.isOpened():
        print(f"✅ Cámara detectada en el índice {i}")
        video.release()
    else:
        print(f"❌ No se encontró cámara en el índice {i}")