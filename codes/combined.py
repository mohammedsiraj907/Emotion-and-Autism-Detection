import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load both models
emotion_model = load_model("emotion_model_ckplus.keras")
autism_model = load_model("autismmodel.keras")

# Label maps
emotion_labels = ['Angry', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness','Surprise']
autism_labels = ["Non-Autistic", "Autistic"]

# Image sizes
emotion_img_size = 96
autism_img_size = 128

# Video capture (choose one)
cap = cv2.VideoCapture("http://192.168.162.152:81/stream")
#cap = cv2.VideoCapture(0)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Face ROI for emotion (grayscale)
        face_gray = gray[y:y+h, x:x+w]
        face_emotion = cv2.resize(face_gray, (emotion_img_size, emotion_img_size))
        face_emotion = np.expand_dims(face_emotion, axis=-1)
        face_emotion = face_emotion.astype("float32") / 255.0
        face_emotion = np.expand_dims(face_emotion, axis=0)

        # Face ROI for autism (color)
        face_color = frame[y:y+h, x:x+w]
        face_autism = cv2.resize(face_color, (autism_img_size, autism_img_size))
        face_autism = face_autism.astype("float32") / 255.0
        face_autism = np.expand_dims(face_autism, axis=0)

        # Emotion prediction
        emotion_preds = emotion_model.predict(face_emotion)[0]
        emotion_label = emotion_labels[np.argmax(emotion_preds)]
        emotion_conf = np.max(emotion_preds) * 100

        # Autism prediction
        autism_pred = autism_model.predict(face_autism)[0][0]
        autism_label = autism_labels[int(autism_pred > 0.5)]

        # Drawing
        color = (0, 255, 0) if autism_label == "Non-Autistic" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Display labels
        cv2.putText(frame, f"{emotion_label} ({emotion_conf:.1f}%)", (x, y-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"{autism_label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Emotion + Autism Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
