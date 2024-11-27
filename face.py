import cv2
import mediapipe as mp


# Function to get face details
def get_face_details(image):
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_details = []
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        # Check if faces are detected
        if results.detections:
            for face_number, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                face_x = int(bbox.xmin * image.shape[1])
                face_y = int(bbox.ymin * image.shape[0])
                face_w = int(bbox.width * image.shape[1])
                face_h = int(bbox.height * image.shape[0])
                face_confidence = detection.score[0]
                
                face_details.append({
                    "face_number": face_number,
                    "face_x": face_x,
                    "face_y": face_y,
                    "face_w": face_w,
                    "face_h": face_h,
                    "face_confidence": face_confidence
                })

                # Draw bounding box and face number on the image
                cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
                cv2.putText(image, f'Face {face_number}', (face_x, face_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(image, f'Confidence: {face_confidence:.2f}', (face_x, face_y + face_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return face_details

if __name__ == "__main__":

    # Open the webcam or video file
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with a video file path

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        face_details = get_face_details(frame)
        print(face_details)

        # Display the frame with face details
        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("face.py has been imported")