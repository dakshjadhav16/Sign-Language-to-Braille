import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.4)
mp_draw = mp.solutions.drawing_utils

DATA_PATH = "../data/signipole_dataset.csv"

def extract_landmarks(frame, apply_blur=False):
    if apply_blur:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5),
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5))

            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

    return np.array(landmarks).flatten() if landmarks else np.zeros(63)

def save_landmarks(label, landmarks):
    df = pd.DataFrame([landmarks])
    df['label'] = label

    if not os.path.exists("../data"):
        os.makedirs("../data")

    df.to_csv(DATA_PATH, mode='a', header=not os.path.exists(DATA_PATH), index=False)
    print(f"✅ Gesture '{label}' Saved Bhai 🚀")

cap = cv2.VideoCapture(0)
gesture = input("Gesture Label Daal Bhai: ")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks(frame, apply_blur=True)

    if np.any(landmarks):
        print("✅ Hand Detected")
        save_landmarks(gesture, landmarks)

    cv2.imshow("Dataset Collector 🔥", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
