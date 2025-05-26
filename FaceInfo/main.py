import cv2
from Facial_emotion_recognition_using_mediapipe.main import getEmotion

image_path = "Hi.jpg"
image = cv2.imread(image_path)
print(getEmotion(image)) 