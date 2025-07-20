import pyttsx3
import cv2
from ultralytics import YOLO

engine = pyttsx3.init()
# Function for text-to-speech
def speech(text):
    engine.say(text)
    engine.runAndWait()

def capture_image():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    
    if not cap.isOpened():
        speech("Error: Could not open webcam.")
        return
    
    # Capture a frame
    speech("Capturing image in  3  2  1")
    ret, frame = cap.read()
    
    if ret:
        #Damage note Detection
        feedback_text = damage_check(frame)
        if feedback_text:
            speech(feedback_text)
    else:
        speech("Error: Could not capture image.")
    cap.release()

class_to_text = {
    "100_new_damaged":"Damaged 100 rupee note detected",
    "100_old_damaged":"Damaged 100 rupee note detected",
    "10_new_damaged": "Damaged 10 rupee note detected",
    "10_old_damaged": "Damaged 10 rupee note detected",
    "20_new_damaged": "Damaged 20 rupee note detected",
    "20_old_damaged": "Damaged 20 rupee note detected",
    "50_new_damaged": "Damaged 50 rupee note detected",
    "50_old_damaged": "Damaged 50 rupee note detected",
    "2":"10 rupee detected",
    "3":"10 rupee detected"
}

def damage_check(image):
    model = YOLO('best (2).pt')
    cv2.namedWindow('YOLOv8 Real-Time Detection', cv2.WINDOW_NORMAL)
    cv2.moveWindow('YOLOv8 Real-Time Detection', 100, 100)
    cv2.resizeWindow('YOLOv8 Real-Time Detection', 800, 600)
    # Use YOLOv8 model to predict objects in the frame
    results = model.track(image, persist=True, conf=0.5, verbose=False)
    feedback_text = None
    if results:
        print("in results")
        annoted_frame = results[0].plot()
        for box in results[0].boxes:
            feedback_text = None
            class_id = int(box.cls)
            print(class_id)
            confidence = float(box.conf)
            # try:
            #     object_id =int(box.id)
            # except:
            #     print("int issues")
            #     return None
            if class_id in range(0,len(model.names)+1):
                print(class_id)
                class_name = model.names[class_id]
                print(class_name)
                if class_name in class_to_text:
                    feedback_text = class_to_text[class_name]
    while True:
        if annoted_frame.any():
            cv2.imshow('YOLOv8 Real-Time Detection',annoted_frame)
        else:
            cv2.imshow('YOLOv8 Real-Time Detection',image)
        # Wait for user response
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return feedback_text  

if __name__ == "__main__":
    capture_image()