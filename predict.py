import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("pet_classifier.h5")

img = cv2.imread("test.jpg")
img = cv2.resize(img,(150,150))
img = img/255.0
img = np.reshape(img,(1,150,150,3))

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")