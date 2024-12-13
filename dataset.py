import os
import cv2
import matplotlib.pyplot as plt

# Define the directory containing the dataset
DATA_DIR = './data'

# Loop through each class directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Construct the full image path
        img_full_path = os.path.join(DATA_DIR, dir_, img_path)
        # Read the image using OpenCV
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Couldn't read image {img_full_path}. Skipping.")
            continue  # This is inside the loop, so it will work correctly
        
        # Convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Display the image using matplotlib
        plt.figure()
        plt.imshow(img_rgb)
        plt.title(f'Class {dir_} - {img_path}')
        plt.show()
        #plt.close()  # Close the figure after displaying it
