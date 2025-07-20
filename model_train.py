import cv2
from ultralytics import YOLO
import pyttsx3
import time
#from picamera2 import Picamera2
# Initialize the pyttsx3 engine once
engine = pyttsx3.init()

# Function for text-to-speech
def speech(text):
    engine.say(text)
    engine.runAndWait()

# Load the YOLO model
model = YOLO('E:/Mini Project/Neckband/neck2.pt')  # Replace with the path to your trained model

# Open the video file or camera
cap = cv2.VideoCapture(0) # Replace with your video file path or camera feed
# picam2 = Picamera2()
# config = picam2.create_preview_configuration()
# picam2.configure(config)

# picam2.start()

# picam2.set_controls({"AfMode": 2})  # 2 = Continuous Autofocus

if not cap.isOpened():
    print("Error: Unable to open video file or camera.")
    exit()

# Create a resizable window
cv2.namedWindow('YOLOv8 Real-Time Detection', cv2.WINDOW_NORMAL)
cv2.moveWindow('YOLOv8 Real-Time Detection', 100, 100)
cv2.resizeWindow('YOLOv8 Real-Time Detection', 800, 600)

# Timer to control audio feedback
last_audio_time = 0  # Initialize the last audio playback time

# Mapping classes to spoken feedback
class_to_text = {
    "500_new": "500 rupee note detected",
    "500_folded": "500 rupee note detected",
    "200_new": "200 rupee note detected",
    "200_new_folded": "200 rupee note detected",
    "100_new": "100 rupee note detected",
    "100_new_folded": "100 rupee note detected",
    "50_new": "50 rupee note detected",
    # Add mappings for other classes as needed
}

totalsum = 0
seen_objects = set()  # Set to track unique (object_id, class_name) pairs
counting_active = False  # Flag to indicate if counting mode is active

while True:
    ret,frame  = cap.read() #picam2.capture_array()  # Capture frame-by-frame
    # # Assuming frame is your input image
    cv2.imshow('YOLOv8 Real-Time Detection',frame)


    #frame_bgr = picam2.capture_array()  # Assuming this captures an RGB frame

    # Convert the frame from RGB to BGR
    #frame = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

    #Display the frame
    # cv2.imshow('YOLOv8 Real-Time Detection', frame)
    # if frame.shape[2] == 4:
    #      frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
    # else:
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB if it's a standard 3-channel image



   # Use YOLOv8 model to predict objects in the frame
    results = model.track(frame, persist=True, conf=0.75, verbose=False)  # Adjust confidence threshold
    current_time = time.time()  # Get the current time

    # Terminate counting mode if no audio feedback for 1 minute
    if counting_active and (current_time - last_audio_time >= 10):
        print("Counting mode terminated due to inactivity\n")
        speech(f"Counting mode terminated, total sum is {totalsum} rupees")
        totalsum = 0
        counting_active = False
        seen_objects.clear()
        print("Cleared seen objects")

    # Annotate the frame with YOLO predictions
    if results:
        annotated_frame = results[0].plot()  # Visualize predictions

        for box in results[0].boxes:  # Iterate over detected boxes
            class_id = int(box.cls)  # Get the class ID
            try:
                object_id = int(box.id)  # Get the unique object ID (ensure it's an integer)
            except:
                print("int issues")
            class_name = model.names[class_id]  # Get class name from ID

            # Combine object ID and class name as a unique key
            unique_object = (object_id, class_name)

            if unique_object not in seen_objects:  # Check if this object-class pair is new
                if class_name in class_to_text:
                    feedback_text = class_to_text[class_name]
                    print(feedback_text)  # Print feedback for debugging
                    speech(feedback_text)  # Trigger audio feedback

                    # Extract the rupee value from the class name
                    content = class_name.split('_')
                    rupees = int(content[0])
                    checker = class_name[-1]

                    if checker != "folded":
                        if not counting_active:
                            print("New counting mode started")
                            speech("New counting mode started")
                            counting_active = True  # Activate counting mode
                        totalsum += rupees
                        print(f"Total sum: {totalsum}\n")
                    else:
                        print("Folded note detected")

                    # Add the object-class pair to the seen objects set
                    seen_objects.add(unique_object)

                    last_audio_time = current_time  # Update the last audio playback time
    else:
        annotated_frame = frame  # Display the frame as-is if no results

    # Show the annotated frame
    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)

    # Press 'q' to quit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()