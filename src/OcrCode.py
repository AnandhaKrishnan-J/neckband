import cv2
import easyocr
from ultralytics import YOLO
from fuzzywuzzy import fuzz
import regex as re
import pyttsx3

engine = pyttsx3.init()
# Function for text-to-speech
def speech(text):
    engine.say(text)
    engine.runAndWait()

def capture_image():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    speech("Capturing image in  3  ...2  ...1")
    # Capture a frame
    ret, frame = cap.read()
    
    if ret:
        #Damage note Detection
        
        feedback_text,currency_detected,damaged_flag = damage_check(frame)
        if currency_detected:
            if damaged_flag:
                speech(feedback_text)
            else:
                # Perform OCR on the captured frame only if the Note is not Damaged
                text_found = extract_text(frame)
                if text_found:
                    if feedback_text:
                        speech(feedback_text)
                        print(feedback_text)
                else:
                    speech("Currency not Valid")
        else:
            speech("Currency not found")
    else:
        print("Error: Could not capture image.")
    
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

def extract_text(image):
    reader = easyocr.Reader(['en'])  # Specify the language
    target_words = {"bank", "reserve", "india"}
    # Perform text detection
    results = reader.readtext(image)
    matched_words = set()
    serial_code_found = False
    pattern = r"(\d{2}[A-Z]){e<=1}(\s+){e<=2}(\d{6}){e<=2}" 
    compiled_pattern = re.compile(pattern)

    # Draw bounding boxes and display detected text
    for (bbox, text, prob) in results:
        # (top_left, top_right, bottom_right, bottom_left) = bbox
        # top_left = tuple(map(int, top_left))
        # bottom_right = tuple(map(int, bottom_right))
        
        # # Draw bounding box
        # cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        
        # # Put text
        # cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
        #             0.8, (0, 0, 255), 2)
        
        print(f'Detected text: {text}, Confidence: {prob:.2f}')

        if prob > 0.1:                              #ADJUST CONFIDENCE HERE
            print("in check :",text)
            for target in target_words:
                text = text.lower()
                similarity = fuzz.ratio(text, target)
                if similarity >= 70:                # Adjust threshold as needed
                    matched_words.add(target)
            if (re.search(compiled_pattern, text)):
                print("pattern match :",text)
                serial_code_found = True
    print(matched_words)
    if matched_words or serial_code_found:
        return True
    else:
        return False



#replace with damaged note messages
class_to_text = {
    "100_new_damaged":"Damaged 100 rupee note detected",
    "100_old_damaged":"Damaged 100 rupee note detected",
    "10_new_damaged": "Damaged 10 rupee note detected",
    "10_old_damaged": "Damaged 10 rupee note detected",
    "20_new_damaged": "Damaged 20 rupee note detected",
    "20_old_damaged": "Damaged 20 rupee note detected",
    "50_new_damaged": "Damaged 50 rupee note detected",
    "50_old_damaged": "Damaged 50 rupee note detected",
    "500_new": "500 rupee note detected",
    "500_folded": "500 rupee note detected",
    "200_new": "200 rupee note detected",
    "200_new_folded": "200 rupee note detected",
    "100_new": "100 rupee note detected",
    "100_new_folded": "100 rupee note detected",
    "50_new": "50 rupee note detected",
    "50_new_folded": "50 rupee note detected",
    "20_new":"20 rupee note detected",
    "20_new_folded":"20 rupee note detected",
    "2":"10 rupee detected",
    "3":"10 rupee detected",
    "13":"500 rupee detected"
}

def damage_check(image):
    model = YOLO('best (2).pt')
    currency_detected = False
    damaged_flag = False
    cv2.namedWindow('YOLOv8 Real-Time Detection', cv2.WINDOW_NORMAL)
    cv2.moveWindow('YOLOv8 Real-Time Detection', 100, 100)
    cv2.resizeWindow('YOLOv8 Real-Time Detection', 800, 600)
    # Use YOLOv8 model to predict objects in the frame
    results = model.track(image, persist=True, conf=0.5, verbose=False)
    annoted_frame = None
    feedback_text = None
    if results:
        print("in results")
        annotated_frame = results[0].plot()
        for box in results[0].boxes:
            class_id = int(box.cls)
            print(class_id)
            confidence = float(box.conf)
            try:
                object_id =int(box.id)
            except:
                print("id not matching")
            if class_id in range(0,len(model.names)+1):
                print(class_id)
                currency_detected = True
                class_name = model.names[class_id]
                if "damaged" in class_name:
                    damaged_flag = True
                print(class_name)
                if class_name in class_to_text:
                    feedback_text = class_to_text[class_name]
                    cv2.imshow('YOLOv8 Real-Time Detection', annotated_frame)
                    # Wait for user response
        while True:
            if annoted_frame:
                cv2.imshow('YOLOv8 Real-Time Detection',annoted_frame)
            else:
                cv2.imshow('YOLOv8 Real-Time Detection',image)
            # Wait for user response
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return feedback_text,currency_detected,damaged_flag
    return None  

if __name__ == "__main__":
    capture_image()
    


