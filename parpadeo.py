import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

# Configuración de Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,  # Permitir múltiples rostros
    refine_landmarks=True,  # Para detección precisa de ojos
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

EAR_THRESHOLD = 0.2
CONSEC_FRAMES = 3

# Diccionario para almacenar contadores por rostro
blink_data = {}

# Función para calcular EAR
def eye_aspect_ratio(landmarks, eye_indices):
    eye = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_indices])
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Captura de video
cap = cv2.VideoCapture(0)

# Inicializamos una lista para los puntos faciales de un rostro conocido
known_face_landmarks = None  # Aquí se guardarán los puntos faciales de un rostro conocido

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)
    
    if result.multi_face_landmarks:
        for face_id, face_landmarks in enumerate(result.multi_face_landmarks):
            # Calcular EAR para ambos ojos
            left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE_INDICES)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Inicializar datos del rostro si no existen
            if face_id not in blink_data:
                blink_data[face_id] = {"frame_counter": 0, "blink_counter": 0}
            
            # Verificar parpadeo
            if avg_ear < EAR_THRESHOLD:
                blink_data[face_id]["frame_counter"] += 1
            else:
                if blink_data[face_id]["frame_counter"] >= CONSEC_FRAMES:
                    blink_data[face_id]["blink_counter"] += 1
                blink_data[face_id]["frame_counter"] = 0

            # Aquí podrías guardar un rostro conocido, por ejemplo si el rostro es el primero detectado
            if known_face_landmarks is None:
                known_face_landmarks = np.array([[landmark.x, landmark.y] for landmark in face_landmarks.landmark])
            
            # Comparar el rostro detectado con el conocido
            if known_face_landmarks is not None:
                detected_landmarks = np.array([[landmark.x, landmark.y] for landmark in face_landmarks.landmark])
                distance = np.linalg.norm(known_face_landmarks - detected_landmarks)  # Distancia Euclidiana

                # Umbral para considerar que es el mismo rostro
                if distance < 0.1:  # Ajusta este valor según sea necesario
                    # Dibujar un rectángulo o un texto si es el mismo rostro
                    cv2.putText(frame, "Rostro conocido detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Dibujar puntos faciales
            h, w, _ = frame.shape
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            # Mostrar contador de parpadeos para cada rostro
            cv2.putText(frame, f"Parpadeos: {blink_data[face_id]['blink_counter']}", 
                        (10, 30 + face_id * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imshow("Seguimiento de Parpadeos y Reconocimiento Facial", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
