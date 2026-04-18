import cv2
import numpy as np
import tensorflow
from pymongo import MongoClient
from urllib.parse import quote_plus
# Load model

USERNAME = quote_plus("admin")
PASSWORD = quote_plus("admin")
model = tensorflow.keras.models.load_model("keras_model.h5", compile=False)
connection_string = f"mongodb+srv://{USERNAME}:{PASSWORD}@civis-ai.gioem70.mongodb.net/?appName=CIVIS-AI"
client = MongoClient(connection_string)
db = client["database"]["database"]

# Load labels
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

lat= 28.68956
long = 77.12101

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame to model input size
    img = cv2.resize(frame, (224, 224))

    # Convert image to numpy array
    img_array = np.asarray(img, dtype=np.float32)

    # Normalize image (Teachable Machine requirement)
    img_array = (img_array / 127.5) - 1

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    class_name = class_names[index]

    # Display result
    label = f"{class_name}: {confidence*100:.2f}%"
    cv2.putText(
        frame,
        label,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Road Damage Detection", frame)
    print(label)

    print(lat, long)
    print(db.find_one({"lat":lat}))
    if lat is not None:
        if class_name == "0 cracks":
            if db.find_one({
                "anomaly":"crack",
                "lat":lat
                }) is None:
                db.insert_one({
                    "anomaly": "crack",
                    "lat":lat,
                    "long":long
                })

        elif class_name == "1 potholes":
            if db.find_one({
                "anomaly":"pothole",
                "lat":lat
                }) is None:
                db.insert_one({
                    "anomaly": "pothole",
                    "lat":lat,
                    "long":long
                })

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
