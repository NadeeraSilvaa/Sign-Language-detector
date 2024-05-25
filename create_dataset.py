import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the hands detector with static_image_mode set to True and a minimum detection confidence of 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory where the dataset is stored
DATA_DIR = './data'

# Initialize lists to store the hand landmarks data and corresponding labels
data = []
labels = []

# Loop over each directory in the dataset directory (each representing a class)
for dir_ in os.listdir(DATA_DIR):
    # Loop over each image file in the current class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Temporary list to store landmarks for the current image
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # Read the image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB color space

        results = hands.process(img_rgb)  # Process the image to detect hand landmarks
        if results.multi_hand_landmarks:  # Check if any hand landmarks are detected
            # Loop over each detected hand's landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                # Loop over each landmark in the hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Get the x-coordinate of the landmark
                    y = hand_landmarks.landmark[i].y  # Get the y-coordinate of the landmark
                    data_aux.append(x)  # Append x-coordinate to the temporary list
                    data_aux.append(y)  # Append y-coordinate to the temporary list

            data.append(data_aux)  # Add the landmark data of the current image to the main data list
            labels.append(dir_)  # Add the corresponding class label to the labels list

# Save the collected data and labels into a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
