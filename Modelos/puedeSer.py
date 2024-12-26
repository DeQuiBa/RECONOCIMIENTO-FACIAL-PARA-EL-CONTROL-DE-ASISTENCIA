import cv2
import mediapipe as mp
import numpy as np
import os

# Configuración de Mediapipe para detección de parpadeos
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,  # Procesar hasta 5 rostros simultáneamente
    refine_landmarks=True,  # Mejora la precisión en los ojos
    min_detection_confidence=0.5,  # Confianza mínima para detección
    min_tracking_confidence=0.5  # Confianza mínima para seguimiento
)

# Umbral de EAR y frames consecutivos necesarios para contar un parpadeo
EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 3
blink_data = {}  # Diccionario para almacenar datos de parpadeos por rostro

# Función para calcular EAR (Eye Aspect Ratio)
def eye_aspect_ratio(landmarks, eye_indices):
    eye = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])  # Distancia vertical entre puntos
    B = np.linalg.norm(eye[2] - eye[4])  # Distancia vertical entre puntos
    C = np.linalg.norm(eye[0] - eye[3])  # Distancia horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# Índices de los puntos de los ojos en el modelo Face Mesh
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Carga de modelo de EigenFaces para reconocimiento facial
dataPath = r"C:\Users\deivi\OneDrive\Escritorio\semana12\data"
imagePaths = os.listdir(dataPath)
print("imagePaths= ", imagePaths)

face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read("modeloEigenFace.xml")

# Configuración de la cámara y clasificador Haar para detección de rostros
cap = cv2.VideoCapture("prueba1.mp4")
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ID correspondiente a "Deivis" (ajustar según cómo se haya entrenado el modelo)
DEIVIS_ID = 1  # Reemplaza con el ID correcto de "Deivis" en tu modelo

while True:
    ret, frame = cap.read()
    if not ret:  # Salir del bucle si no hay más frames
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir frame a escala de grises
    auxFrame = gray.copy()

    # Detectar rostros en el frame actual
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Procesar rostro detectado para reconocimiento facial
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        label, confidence = face_recognizer.predict(rostro)

        # Verificar si el rostro es reconocido como "Deivis"
        if label == DEIVIS_ID:  # Si el ID coincide con el de "Deivis"
            name = "Deivis"
            color = (0, 255, 0)  # Verde si es "Deivis"
        else:  # Si no es reconocido
            name = "Desconocido"
            color = (0, 0, 255)  # Rojo si no es "Deivis"

        # Dibujar el recuadro y el nombre sobre el rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Recortar rostro para análisis de parpadeos
        face_frame = frame[y:y + h, x:x + w]
        rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

        # Detectar puntos faciales con Mediapipe
        result = face_mesh.process(rgb_face_frame)
        if result.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(result.multi_face_landmarks):
                # Calcular EAR para ambos ojos
                left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_INDICES)
                right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_INDICES)
                avg_ear = (left_ear + right_ear) / 2.0

                # Inicializar datos de parpadeos si es un rostro nuevo
                if face_id not in blink_data:
                    blink_data[face_id] = {"frame_counter": 0, "blink_counter": 0}

                # Contar parpadeos basados en EAR
                if avg_ear < EAR_THRESHOLD:
                    blink_data[face_id]["frame_counter"] += 1
                else:
                    if blink_data[face_id]["frame_counter"] >= CONSEC_FRAMES:
                        blink_data[face_id]["blink_counter"] += 1
                    blink_data[face_id]["frame_counter"] = 0

                # Mostrar contador de parpadeos en pantalla
                cv2.putText(frame, f"Parpadeos: {blink_data[face_id]['blink_counter']}",
                            (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Mostrar el frame procesado
    cv2.imshow("Reconocimiento Facial y Detección de Parpadeos", frame)

    # Salir del bucle al presionar 'ESC'
    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
