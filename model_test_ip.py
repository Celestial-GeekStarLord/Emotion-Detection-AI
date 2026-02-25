import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Configuration
MODEL_PATH = "C:\Emotion_Detection\ML_Models\Emotion_Recognition_Model.h5"
IP_WEBCAM_URL = "http://192.168.18.4:8080/video" # Check if your app uses /video or /shot.jpg
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 2. Load Model and Face Detector
print("Loading model...")
model = load_model(MODEL_PATH)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Connect to IP Webcam
cap = cv2.VideoCapture(IP_WEBCAM_URL)

if not cap.isOpened():
    print("Error: Could not open video stream. Check IP address and ensure app is running.")
    exit()

print("Connection established. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Pre-process the face for the model
        roi_gray = frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48)) # Matching your model input size
        
        # If your model expects 3 channels (Transfer Learning MobileNet)
        roi_input = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB)
        
        # Expand dims for batch size: (1, 48, 48, 3)
        roi_input = np.expand_dims(roi_input, axis=0) 

        # 4. Predict
        prediction = model.predict(roi_input, verbose=0)
        label = EMOTIONS[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display label
        label_text = f"{label} ({confidence:.1f}%)"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 5. Show Result
    cv2.imshow('IP Webcam Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()