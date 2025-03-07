### File: src/mediapipe_hand.py
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def extract_landmarks(frame, apply_blur=False):
    if apply_blur:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    landmarks = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Yaha Magic Line Lagayi 🔥
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=5),  # Green Lines
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5))  # Red Dots
            
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y, landmark.z])

    return np.array(landmarks).flatten() if landmarks else np.zeros(63)

cap = cv2.VideoCapture(0)  # Webcam Open 🔥

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks(frame, apply_blur=False)
    
    if np.any(landmarks):
        print("✅ Hand Detected")
        print("Landmarks Shape:", landmarks.shape)

    cv2.imshow("Hand Tracking Skeleton ✋🧠", frame)  # Skeleton Dikh Jayega 🔥

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# ### File: src/dataset_loader.py
# import os
# import cv2
# import numpy as np
# import pickle
# from mediapipe_hand import extract_landmarks

# DATASET_PATH = "dataset/"
# LABELS = ['A', 'B', 'C', 'Hello', 'Thanks']

# def collect_data(apply_blur=False):
#     data = {'landmarks': [], 'labels': []}
#     for label in LABELS:
#         path = os.path.join(DATASET_PATH, label)
#         if not os.path.exists(path):
#             continue
#         for file in os.listdir(path):
#             img = cv2.imread(os.path.join(path, file))
#             landmarks = extract_landmarks(img, apply_blur=apply_blur)
#             if landmarks.sum() != 0:
#                 data['landmarks'].append(landmarks)
#                 data['labels'].append(label)

#     with open("dataset/gesture_data.pkl", "wb") as f:
#         pickle.dump(data, f)

# ### File: src/augmentation.py
# import cv2
# import random
# import os

# DATASET_PATH = "dataset/"

# # Apply Augmentation
# def augment_image(image):
#     if random.random() > 0.5:
#         image = cv2.flip(image, 1)  # Horizontal Flip
#     if random.random() > 0.5:
#         image = cv2.GaussianBlur(image, (5, 5), 0)  # Gaussian Blur
#     return image

# def apply_augmentation():
#     for label in os.listdir(DATASET_PATH):
#         label_path = os.path.join(DATASET_PATH, label)
#         if os.path.isdir(label_path):
#             for file in os.listdir(label_path):
#                 file_path = os.path.join(label_path, file)
#                 img = cv2.imread(file_path)
#                 if img is not None:
#                     aug_img = augment_image(img)
#                     aug_file_path = file_path.replace(".jpg", "_aug.jpg")
#                     cv2.imwrite(aug_file_path, aug_img)

# ### File: src/utils.py
# import os
# import cv2

# def create_dataset_dirs(labels):
#     for label in labels:
#         path = os.path.join("dataset", label)
#         os.makedirs(path, exist_ok=True)
