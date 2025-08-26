# import cv2
# import numpy as np
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import img_to_array

# # Load the model saved in Keras format
# model = keras.models.load_model('model/lumpy_skin_model.keras')

# # Start webcam
# cap = cv2.VideoCapture(0)

# def predict_frame(frame):
#     frame_resized = cv2.resize(frame, (224, 224))
#     img_array = img_to_array(frame_resized) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     pred = model.predict(img_array)[0][0]
#     return "Infected" if pred >= 0.5 else "Healthy", pred

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     label, confidence = predict_frame(frame)
#     color = (0, 0, 255) if label == "Infected" else (0, 255, 0)
#     text = f"{label} ({confidence*100:.2f}%)"

#     cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#     cv2.imshow('LSD Real-Time Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = keras.models.load_model("model/lumpy_skin_model.keras")

# Start webcam
cap = cv2.VideoCapture(0)

frame_count = 0
last_label, last_conf = "Healthy", 1.0

def predict_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    img_array = img_to_array(frame_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]

    if preds.shape[0] == 1:  # Sigmoid output
        conf = float(preds[0])
        label = "Infected" if conf >= 0.5 else "Healthy"
        confidence = conf if label == "Infected" else 1 - conf
    else:  # Softmax output
        pred_class = np.argmax(preds)
        confidence = float(preds[pred_class])
        label = "Infected" if pred_class == 1 else "Healthy"

    return label, confidence

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 10 == 0:  # Predict every 10 frames
        last_label, last_conf = predict_frame(frame)

    color = (0, 0, 255) if last_label == "Infected" else (0, 255, 0)
    text = f"{last_label} ({last_conf*100:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("LSD Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
