import cv2
import numpy as np
from mediapipe import solutions

def process_hands(frame):
    hand_module = solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_module.process(frame_rgb)
    hand_landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Append landmark coordinates in (x, y, z)
            hand_landmarks.append([(lm.x, lm.y, lm.z) for lm in hand_landmark.landmark])
    return hand_landmarks

def draw_hand_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape
    for hand in hand_landmarks:
        # Draw landmarks and connections
        for idx, landmark in enumerate(hand):
            x, y, z = int(landmark[0] * width), int(landmark[1] * height), landmark[2]
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            # Highlight key points: WRIST and PINKY_TIP
            if idx in [0, 17]:  # WRIST and PINKY_TIP
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    return frame

def main():
    cap = cv2.VideoCapture(0)
    hand_module = solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hand_landmarks = process_hands(frame)
        no_of_hands = len(hand_landmarks)
        
        # Draw detected hands and landmarks
        frame = draw_hand_landmarks(frame, hand_landmarks)

        # Display the number of hands detected
        cv2.putText(frame, f"Hands Detected: {no_of_hands}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show output
        cv2.imshow('Hand Tracking', frame)

        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
