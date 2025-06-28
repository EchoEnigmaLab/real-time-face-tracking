import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from google.colab import files
from matplotlib import pyplot as plt
uploaded = files.upload()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.h5')  
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
image_path = list(uploaded.keys())[0] 
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found!")

else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y + h, x:x + w]

        roi_color = image[y:y + h, x:x + w]

        roi_resized = cv2.resize(roi_gray, (48, 48))

        roi_resized = roi_resized.astype('float32') / 255

        roi_resized = img_to_array(roi_resized)

        roi_resized = np.expand_dims(roi_resized, axis=0)

        emotion_pred = emotion_model.predict(roi_resized)

        max_index = np.argmax(emotion_pred[0])

        predicted_emotion = emotion_labels[max_index]
        cv2.putText(image, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.axis('off')

    plt.show()

