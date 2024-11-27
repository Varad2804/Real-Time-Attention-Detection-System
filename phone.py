import torch
import cv2

# Load YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'yolov7', pretrained=True)

# Load COCO names (assuming 'coco.names' file exists)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (0 for webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam or video
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, channels = frame.shape

    # Convert frame to RGB (since YOLO expects RGB format)
    img = frame[..., ::-1]  # Convert BGR to RGB

    # Perform inference with YOLOv7
    results = model(img)

    # Extract the results as pandas dataframe
    detections = results.pandas().xywh[0]  # Get the first image's detections as a dataframe

    # Initialize a counter for phones detected
    phone_count = 0

    # Process each detection
    for i, detection in detections.iterrows():
        # Extract bounding box coordinates (x_center, y_center, width, height)
        x_center = detection['xcenter']
        y_center = detection['ycenter']
        w = detection['width']
        h = detection['height']
        
        # Confidence and class_id
        confidence = detection['confidence']
        class_id = int(detection['class'])
        label = detection['name']

        # Only process the "phone" class (or whatever class you're interested in)
        if confidence > 0.5 and label.lower() == 'cell phone':  # Adjust class name as necessary
            # Increment the phone counter
            phone_count += 1

            # Convert to top-left corner coordinates for OpenCV
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)

            # Print the coordinates of the detected phone
            print(f"Phone detected with coordinates: x={x}, y={y}, w={w}, h={h}")

            # Draw a bounding box around the detected phone
            cv2.rectangle(frame, (x, y), (x + int(w), y + int(h)), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Print total number of phones detected in the current frame
    print(f"Total phones detected: {phone_count}")

    # Display the resulting frame with bounding boxes
    cv2.imshow("Frame", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

