import cv2
import os
import time

# Ruta de las imágenes de entrenamiento
dataPath = r"C:\Users\deivi\OneDrive\Escritorio\semana12\data"
imagePaths = os.listdir(dataPath)
print("imagePaths= ", imagePaths)

# Cargar el modelo de EigenFace
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read("modeloEigenFace.xml")

# Captura de video
cap = cv2.VideoCapture("Prueba4.mp4")

# Clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Variables para calcular métricas
TP = 0  # True Positives
TN = 0  # True Negatives
FP = 0  # False Positives
FN = 0  # False Negatives
times = []  # Lista para almacenar tiempos de procesamiento por rostro
total_faces = 0  # Total de rostros procesados

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces: 
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        # Calcular tiempo de procesamiento
        start_time = time.time()

        result = face_recognizer.predict(rostro)
        
        # Calcular tiempo de procesamiento por rostro
        end_time = time.time()
        times.append(end_time - start_time)

        total_faces += 1

        # Obtener la confianza y determinar el nombre
        confidence = result[1]
        if confidence < 5700 and result[0] < len(imagePaths):
            predicted_name = imagePaths[result[0]]  # Nombre predicho correctamente
        else:
            predicted_name = "No identificado"  # Si la confianza es alta o el índice es inválido

        # Configuración de colores y texto
        fontScale = 0.6  # Escala más pequeña para el texto
        color = (0, 255, 0) if predicted_name != "No identificado" else (0, 0, 255)  # Verde o rojo
        text = f"{predicted_name} - Valor de ajuste: {confidence:.2f}"

        # Mostrar el texto en el marco
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, 2, cv2.LINE_AA)

        # Dibujar el rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Actualizar métricas
        if confidence < 5700:  # Confianza aceptable
            if predicted_name != "No identificado":
                TP += 1
            else:
                FP += 1
        else:  # Confianza no aceptable
            FN += 1

    # Mostrar el video con los textos superpuestos
    cv2.imshow("frame", frame)
    k = cv2.waitKey(50)
    if k == 27:  # Presionar 'Esc' para salir
        break

# Calcular TPRF y TPPR correctamente
TPRF = (TP / (TP + FP + FN)) * 100 if (TP + FP + FN) > 0 else 0  # Tasa de exactitud basada en predicciones correctas
TPPR = sum(times) / total_faces if total_faces > 0 else 0  # Tiempo Promedio de Procesamiento por Rostro

# Imprimir resultados finales
print("\n--- Resultados finales ---")
print(f"TP (True Positives): {TP}")
print(f"FP (False Positives): {FP}")
print(f"FN (False Negatives): {FN}")
print(f"TPRF (Tasa de Exactitud): {TPRF:.2f}%")
print(f"TPPR (Tiempo Promedio de Procesamiento por Rostro): {TPPR:.4f} segundos")


cap.release()
cv2.destroyAllWindows()
