import csv
import copy
import itertools

import cv2 as cv
import numpy as np
import mediapipe as mp
from Facial_emotion_recognition_using_mediapipe.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

use_brect = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

keypoint_classifier = KeyPointClassifier()


# Read labels
with open('Facial_emotion_recognition_using_mediapipe/model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

mode = 0

def getEmotion(image):
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    if results.multi_face_landmarks is not None:
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Bounding box calculation
            brect = calc_bounding_rect(debug_image, face_landmarks)

            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, face_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)

            #emotion classification
            facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
            landmarks = face_landmarks.landmark

            # Eye blink
            left_eye_ratio = get_eye_ratio(landmarks, 159, 145)
            right_eye_ratio = get_eye_ratio(landmarks, 386, 374)
            avg_eye_ratio = (left_eye_ratio + right_eye_ratio) / 2
            blink_status = "Blink" if avg_eye_ratio < 0.015 else "Eyes Open"

            # Eye movement (gaze)
            iris_pos = get_iris_position(landmarks, 33, 133, 468)
            if iris_pos < 0.35:
                gaze = "Looking Left"
            elif iris_pos > 0.65:
                gaze = "Looking Right"
            else:
                gaze = "Looking Center"
            
            return [keypoint_classifier_labels[facial_emotion_id], gaze, blink_status]
    return "No Face Detected"
    

def get_eye_ratio(landmarks, top_idx, bottom_idx):
    return abs(landmarks[top_idx].y - landmarks[bottom_idx].y)

def get_iris_position(landmarks, left_idx, right_idx, iris_idx):
    eye_left = landmarks[left_idx]
    eye_right = landmarks[right_idx]
    iris = landmarks[iris_idx]
    return (iris.x - eye_left.x) / abs(eye_right.x - eye_left.x)

def to_pixel(lm):
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)