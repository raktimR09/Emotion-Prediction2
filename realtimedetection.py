import cv2
from keras.models import model_from_json
import numpy as np

# Load model architecture and weights
with open("emotiondetector.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")


# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)  # Set width to 720
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 640) # Set height to 640

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_image = gray[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        cv2.putText(im, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow("Output", im)

    if cv2.waitKey(1) == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
