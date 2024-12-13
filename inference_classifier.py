import cv2
import pickle
import mediapipe as mp
import numpy as np

# Load the model from file
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels
labels_dict = {0: 'pataka', 1: 'ardhapatka', 2: 'tripataka'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Initialize variables for drawing bounding box and text
    x1, y1, x2, y2 = 0, 0, 0, 0
    predicted_character = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmarks for prediction
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        
        if x_ and y_:
            # Calculate bounding box coordinates
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

    # Draw bounding box and text
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.putText(frame, predicted_character, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
