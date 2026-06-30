import cv2
import numpy as np
from keras.models import load_model

# =========================
# LOAD MODELS
# =========================
emotion_model = load_model("emotion_model_ckplus.keras")
autism_model = load_model("autismmodel.keras")

# =========================
# LABELS
# =========================
emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
autism_labels = ["Non-Autistic", "Autistic"]

# =========================
# IMAGE SIZES
# =========================
emotion_img_size = 96
autism_img_size = 128

# =========================
# IP WEBCAM STREAM URL
# =========================
#Replace with your phone IP
# ip_webcam_url = "https://192.168.137.101:8080/stream"

# cap = cv2.VideoCapture(ip_webcam_url, cv2.CAP_FFMPEG)
# =========================
# VIDEO CAPTURE (LAPTOP CAMERA)
# =========================
cap = cv2.VideoCapture(0)

# Optional: reduce lag
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(3, 640)   # width
cap.set(4, 480)   # height

# =========================
# FACE DETECTION
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        # =========================
        # EMOTION MODEL (GRAYSCALE)
        # =========================
        face_gray = gray[y:y+h, x:x+w]
        face_emotion = cv2.resize(face_gray, (emotion_img_size, emotion_img_size))
        face_emotion = face_emotion.astype("float32") / 255.0
        face_emotion = np.expand_dims(face_emotion, axis=-1)
        face_emotion = np.expand_dims(face_emotion, axis=0)

        emotion_preds = emotion_model.predict(face_emotion, verbose=0)[0]
        emotion_label = emotion_labels[np.argmax(emotion_preds)]
        emotion_conf = np.max(emotion_preds) * 100

        # =========================
        # AUTISM MODEL (COLOR)
        # =========================
        face_color = frame[y:y+h, x:x+w]
        face_autism = cv2.resize(face_color, (autism_img_size, autism_img_size))
        face_autism = face_autism.astype("float32") / 255.0
        face_autism = np.expand_dims(face_autism, axis=0)

        autism_pred = autism_model.predict(face_autism, verbose=0)[0][0]
        autism_label = autism_labels[int(autism_pred > 0.5)]

        # =========================
        # DRAWING BOX + TEXT
        # =========================
        color = (0, 255, 0) if autism_label == "Non-Autistic" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            f"{emotion_label} ({emotion_conf:.1f}%)",
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"{autism_label}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # =========================
    # DISPLAY WINDOW
    # =========================
    cv2.imshow("Emotion + Autism Detection (IP Webcam)", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()