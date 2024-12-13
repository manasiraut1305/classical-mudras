import os
import cv2

# Define the directory to store the dataset
DATA_DIR = './data'

# Create the directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 10

# Try different camera indices if necessary
camera_index = 0  # You can try changing this index if the default doesn't work

# Open the video capture
cap = cv2.VideoCapture(camera_index)

# Check if the video capture opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video capture device at index {camera_index}.")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}. Press "q" to start.')

    # Wait for the user to be ready
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.putText(frame, 'Ready? Press "q" to start!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Collect dataset_size number of images for each class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1

    print(f'Data collection for class {j} completed.')

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
