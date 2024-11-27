import cv2
import mediapipe as mp
import numpy as np
from posef import get_pose_para
from face import get_face_details
from hand import process_hands
from hand import draw_hand_landmarks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Open the webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with a video file path
# Define the fixed mapping for categorical variables (same as during training)
label_mapping = {'forward': 0, 'down': 1, 'left': 2, 'right': 3}

# Load the pre-trained model weights and scaler
scaler = joblib.load('scaler.pkl')
weights_file = 'model2.weights(1).h5'

# Step 1: Define the same model architecture as during training
# Ensure the input shape matches the shape used during training
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(10,)))  # Input layer (adjust 10 to match number of features)
model.add(Dense(64, activation='relu'))  # Hidden layer 1
model.add(Dense(32, activation='relu'))  # Hidden layer 2
model.add(Dense(4, activation='softmax'))  # Output layer with 4 classes (for 'forward', 'down', 'left', 'right')

# Load the trained model weights
model.load_weights(weights_file)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # print("hey there\n")
    face_details = get_face_details(frame)
    pose,pose_x,pose_y=get_pose_para(frame)
    hand_landmarks = process_hands(frame)
    no_of_hands = len(hand_landmarks)
    frame = draw_hand_landmarks(frame, hand_landmarks)
    cv2.putText(frame, f"Hands Detected: {no_of_hands}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    print(f"Number of hands:{no_of_hands}")
    print(face_details)
    print(f'pose_x:{pose_x},pose_y:{pose_y},pose:{pose}')
    no_of_face=1
    # print(face_details[0]['face_x'])
    sample_input=[1,face_details[0]['face_x'],face_details[0]['face_y'],face_details[0]['face_w'],face_details[0]['face_h'],face_details[0]['face_confidence'],no_of_hands,pose,pose_x,pose_y]
    # Step 3: Apply the label mapping to the categorical column (index 7)
    sample_input[7] = label_mapping[sample_input[7]]  # 'down' -> 1

    # Step 4: Preprocess the input data
    # Reshape the input sample to be 2D (1 sample, n features) and scale it
    sample_input_scaled = scaler.transform(np.array(sample_input).reshape(1, -1))  # Scaling the input

    # Step 5: Make predictions
    predictions = model.predict(sample_input_scaled)
    predicted_class = np.argmax(predictions, axis=1)  # Get the class with the highest probability

    # # Step 6: Convert the predicted class back to the corresponding label
    # predicted_label = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_class[0])]

    # Step 7: Print the predicted class
    print(f"Predicted class: {predicted_class}")
    # Display the frame with face details
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
