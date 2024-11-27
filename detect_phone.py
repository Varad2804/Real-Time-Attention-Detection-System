import cv2
from ultralytics import YOLO

# Load your YOLO model (adjust path if necessary)
model = YOLO('yolov5su.pt')  # Change this to the path of your trained YOLO model

def detect_phone_on_frame(model, frame):
    # Run the frame through the model to get the results
    results = model(frame)

    # Initialize variables to store the phone detection details
    phone_present = 0  # Default value, if no phone is detected
    phone_x, phone_y, phone_w, phone_h = None, None, None, None
    phone_conf = None

    # Extract results from the model
    for result in results:
        # Extract the boxes and class names
        boxes = result.boxes.xyxy  # bounding boxes as xyxy format
        confidences = result.boxes.conf  # confidence scores for each detected object
        class_ids = result.boxes.cls  # class ids of detected objects
        labels = result.names  # class names for the detected objects
        
        # Iterate over all detections
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            # Convert the class_id tensor to a scalar integer
            class_id = int(class_id.item())

            # Check if the detected class is 'cell phone' (adjust the label accordingly)
            if labels[class_id] == 'cell phone':  # Adjust this for your model's class names
                phone_present = 1  # Phone detected
                # Extract the bounding box (x, y, width, height) and confidence
                x1, y1, x2, y2 = box
                phone_x, phone_y = x1.item(), y1.item()  # Upper-left corner
                phone_w, phone_h = (x2 - x1).item(), (y2 - y1).item()  # Width and height
                phone_conf = conf.item()  # Confidence score

                break  # Break if the phone is found (no need to check further)

    return phone_present, phone_x, phone_y, phone_w, phone_h, phone_conf


def main():
    # Initialize webcam capture (0 for the default webcam, change if needed)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    while True:
        # Capture each frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Detect the phone in the current frame
        phone_present, phone_x, phone_y, phone_w, phone_h, phone_conf = detect_phone_on_frame(model, frame)

        # Output the results
        print(f"Phone Present: {phone_present}")
        if phone_present == 1:
            print(f"Phone Bounding Box: x={phone_x}, y={phone_y}, w={phone_w}, h={phone_h}")
            print(f"Phone Confidence: {phone_conf}")

            # Draw bounding box and confidence score on the frame
            cv2.rectangle(frame, (int(phone_x), int(phone_y)), (int(phone_x + phone_w), int(phone_y + phone_h)), (0, 255, 0), 2)
            cv2.putText(frame, f"Phone: {phone_conf:.2f}", (int(phone_x), int(phone_y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Show the frame with detections
        cv2.imshow('Phone Detection - Webcam', frame)

        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
