import cv2
import mediapipe as mp
import numpy as np
from posef import get_pose_para
from face import get_face_details
from hand import process_hands
from hand import draw_hand_landmarks
# Open the webcam or video file
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with a video file path

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

    # Display the frame with face details
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()