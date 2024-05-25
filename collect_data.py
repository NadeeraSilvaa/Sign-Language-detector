import os
import cv2

# Define the directory where the dataset will be stored
DATA_DIR = './data'

# Check if the directory exists, if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Set the number of classes and the size of the dataset per class
number_of_classes = 3
dataset_size = 100

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Loop over the number of classes
for j in range(number_of_classes):
    # Create a subdirectory for each class if it doesn't already exist
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    # Notify the user that data collection for the current class is starting
    print('Collecting data for class {}'.format(j))

    done = False
    # Wait for the user to signal they are ready to start collecting data
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        # Display instructions on the frame
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame in a window
        # Wait for the user to press 'q' to start data collection
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    # Collect the specified number of images for the current class
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the webcam
        cv2.imshow('frame', frame)  # Show the frame in a window
        cv2.waitKey(25)  # Wait for 25 milliseconds
        # Save the frame as an image file in the corresponding class directory
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1  # Increment the counter

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
