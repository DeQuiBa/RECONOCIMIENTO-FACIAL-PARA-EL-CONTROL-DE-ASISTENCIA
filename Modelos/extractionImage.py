import cv2
import os
import imutils
import numpy as np

# Nombre de la persona
firstPerson = "Fabian"
dataPath = r"C:\Users\deivi\OneDrive\Escritorio\semana12\data"  # Ruta de datos
personPath = os.path.join(dataPath, firstPerson)

# Crear la carpeta si no existe
if not os.path.exists(personPath):
    print("Creando carpeta de datos para la persona:", personPath)
    os.makedirs(personPath)

# Abrir el video
screenshot = cv2.VideoCapture("Fabian.mp4")

# Cargar el clasificador Haar para la detección de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Verificar si el clasificador se carga correctamente
if faceClassif.empty():
    print("Error al cargar el clasificador de rostros")
    exit()

# Contador de imágenes
count = 0

# Lista para almacenar rostros y etiquetas
faces_data = []
labels = []

# Etiqueta para la persona (solo 1 persona en este caso, "Deivis")
label = 1

# Extraer imágenes de rostro
while True:
    ret, frame = screenshot.read()
    if not ret:
        break  # Termina si no hay más frames

    # Redimensionar la imagen
    frame = imutils.resize(frame, width=640)

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = faceClassif.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No se detectaron rostros en el frame.")
    else:
        for (x, y, w, h) in faces:
            # Filtrar rostros pequeños
            if w < 100 or h < 100:
                continue

            # Dibujar un rectángulo alrededor del rostro
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extraer la región del rostro
            cara = frame[y:y + h, x:x + w]

            # Redimensionar el rostro
            cara = cv2.resize(cara, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Guardar la imagen del rostro
            cv2.imwrite(f"{personPath}/rostro_{count}.jpg", cara)
            count += 1

            # Agregar el rostro y la etiqueta a la lista
            faces_data.append(cara)
            labels.append(label)

    # Mostrar el video con rostros detectados
    cv2.imshow("frame", frame)

    # Salir si se presiona la tecla ESC o si se han capturado suficientes rostros
    k = cv2.waitKey(1)
    if k == 27 or count >= 400:  # 27 es la tecla ESC
        break

# Liberar el video y cerrar ventanas
screenshot.release()
cv2.destroyAllWindows()

