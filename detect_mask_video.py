import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time

# --- CONFIGURATION ---
MODEL_PATH = "mask_detector.h5"
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

# --- LOAD MODELS ---
print("[INFO] Loading face detector...")
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

if face_cascade.empty():
    print("[ERROR] Could not load face cascade! Check the path.")
    exit()

print("[INFO] Loading mask detector model...")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    exit()

# --- START VIDEO STREAM ---
print("[INFO] Starting video stream...")
# Try 0 for default webcam. If you have an external cam, try 1.
vs = cv2.VideoCapture(0) 
time.sleep(2.0) # Allow camera to warm up

while True:
    # 1. Grab frame
    ret, frame = vs.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break
        
    # Mirror effect for natural feel
    frame = cv2.flip(frame, 1) 
    
    # 2. Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60)
    )

    # 3. Loop over each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess the face for MobileNetV2
        try:
            # Convert from BGR (OpenCV) to RGB (Keras)
            face_input = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_input = cv2.resize(face_input, (224, 224))
            face_input = img_to_array(face_input)
            face_input = preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)

            # 4. Predict
            (withoutMask, mask) = model.predict(face_input, verbose=0)[0]

            # 5. Determine Label and Color
            # logic: if mask probability > withoutMask, then "Mask"
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255) # Green vs Red
            
            # Format label with probability
            prob = max(mask, withoutMask) * 100
            label_text = "{}: {:.2f}%".format(label, prob)

            # 6. Draw box and label
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
        except Exception as e:
            print(f"Error processing face: {e}")

    # Show the frame
    cv2.imshow("Face Mask Detector", frame)
    
    # Press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
vs.release()
cv2.destroyAllWindows()

            