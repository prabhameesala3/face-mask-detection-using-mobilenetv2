# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# # -----------------------------
# # 1. Load model & class labels
# # -----------------------------
# MODEL_PATH = "model/face_mask_mobilenetv2.h5"
# model = load_model(MODEL_PATH)

# # Make sure these are in same order as class_indices used during training
# # Example: {'with_mask': 0, 'without_mask': 1}
# CLASS_NAMES = ["with_mask", "without_mask"]

# # For face detection, we use OpenCV's pre-trained Haar Cascade
# face_cascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# )

# # -----------------------------
# # 2. Start webcam
# # -----------------------------
# cap = cv2.VideoCapture(0)  # 0 = default camera

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(60, 60)
#     )

#     for (x, y, w, h) in faces:
#         # Extract face region
#         face = frame[y:y+h, x:x+w]
#         # Preprocess for MobileNetV2
#         face_resized = cv2.resize(face, (224, 224))
#         face_array = np.expand_dims(face_resized, axis=0)
#         face_array = preprocess_input(face_array.astype("float32"))

#         # Predict
#         preds = model.predict(face_array, verbose=0)[0]
#         class_id = np.argmax(preds)
#         label = CLASS_NAMES[class_id]
#         confidence = preds[class_id]

#         # Choose color: green for mask, red for no mask
#         color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)

#         # Draw rectangle & label
#         cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#         text = f"{label}: {confidence*100:.1f}%"
#         cv2.putText(frame, text, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     cv2.imshow("Face Mask Detection - TL-MaskNet", frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model/face_mask_mobilenetv2.h5"
model = load_model(MODEL_PATH)

IMG_SIZE = 224

# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # Prediction
        pred = model.predict(face)[0][0]

        if pred > 0.5:
            label = "without_mask"
            color = (0, 0, 255)       # RED
            confidence = pred * 100
        else:
            label = "with_mask"
            color = (0, 255, 0)       # GREEN
            confidence = (1 - pred) * 100

        text = f"{label}: {confidence:.2f}%"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detection - TL-MaskNet", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
