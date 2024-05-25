import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model from the pickle file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the hands detector with static_image_mode set to True and a minimum detection confidence of 0.3
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the label dictionary to map the predicted labels to actual characters
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

# Start the video capture loop
while True:
    data_aux = []  # Temporary list to store landmarks data for the current frame
    x_ = []  # List to store x-coordinates of the landmarks
    y_ = []  # List to store y-coordinates of the landmarks

    ret, frame = cap.read()  # Capture a frame from the webcam

    H, W, _ = frame.shape  # Get the dimensions of the frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB color space

    results = hands.process(frame_rgb)  # Process the frame to detect hand landmarks

    if results.multi_hand_landmarks:  # Check if any hand landmarks are detected
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Extract and normalize the hand landmark coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x  # Get the x-coordinate of the landmark
                y = hand_landmarks.landmark[i].y  # Get the y-coordinate of the landmark
                data_aux.append(x)  # Append x-coordinate to the temporary list
                data_aux.append(y)  # Append y-coordinate to the temporary list
                x_.append(x)  # Append x-coordinate to the list for bounding box calculation
                y_.append(y)  # Append y-coordinate to the list for bounding box calculation

        # Calculate the bounding box around the detected hand
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Predict the character using the trained model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]  # Map the prediction to the actual character

        # Draw the bounding box and predicted character on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)  # Display the frame with the annotations
    cv2.waitKey(1)  # Wait for 1 millisecond before processing the next frame

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
