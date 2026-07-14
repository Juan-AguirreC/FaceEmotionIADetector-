import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# CARGAR MODELO
# ==========================================
model = tf.keras.models.load_model(
    "emotion_model_5classes_surprised.keras"
)

# ==========================================
# CLASES
# ==========================================
emotion_labels = [
    'angry',
    'happy',
    'neutral',
    'sad',
    'surprised'
]

# ==========================================
# DETECTOR DE ROSTROS
# ==========================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Verificar carga correcta
if face_cascade.empty():
    print("Error cargando Haar Cascade")
    exit()

# ==========================================
# ABRIR CÁMARA
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Presiona Q para salir")

# ==========================================
# LOOP PRINCIPAL
# ==========================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Error leyendo cámara")
        break

    # ==========================================
    # ESCALA DE GRISES
    # ==========================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # DETECCIÓN DE ROSTROS
    # ==========================================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ==========================================
    # PROCESAR CADA ROSTRO
    # ==========================================
    for (x, y, w, h) in faces:

        # Rectángulo
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Extraer rostro
        face = gray[y:y+h, x:x+w]

        # Redimensionar
        face = cv2.resize(face, (48, 48))

        # Normalizar
        face = face.astype("float32") / 255.0

        # Shape: (1,48,48,1)
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        # Predicción
        prediction = model.predict(face, verbose=0)

        emotion_index = np.argmax(prediction)

        emotion = emotion_labels[emotion_index]

        confidence = np.max(prediction) * 100

        # Texto
        text = f"{emotion} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Mostrar frame
    cv2.imshow("Emotion Detection", frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================================
# LIBERAR
# ==========================================
cap.release()
cv2.destroyAllWindows()