import cv2
import pytesseract
from fuzzywuzzy import fuzz
import numpy as np


def capture_image():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Capture a frame
    ret, frame = cap.read()
    
    if ret:
        # Show the captured frame
        cv2.imshow("Captured Image", frame)
        print("Press 'q' to exit and process the image.")
        
        # Wait for user response
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        # Perform OCR on the captured frame
        extract_text(frame)
    else:
        print("Error: Could not capture image.")
    
    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()

def preprocess_image(image):
    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Adaptive Thresholding to enhance contrast
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    
    # Apply Morphological Transformations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    return processed

def extract_text(image):
    target_words = {"bank", "reserve", "india"}
    # Convert the image to RGB
    #image = preprocess_image(frame)

    # Perform OCR using Tesseract
    extracted_text = pytesseract.image_to_string(image).lower()
    
    print("Extracted Text:")
    print(extracted_text)
    
    # Check for approximate matches using fuzzy matching
    matched_words = set()
    
    count_word = 0
    for target in target_words:
        similarity = fuzz.ratio(extracted_text, target)
        if similarity >= 80:  # Adjust threshold as needed
            matched_words.add(target)
            count_word = count_word + 1
    try:
        if (count_word/len(matched_words)):
            print(f"Matched Words (Fuzzy Matching): {matched_words}")
    except:
        print("No target words found.") 

if __name__ == "__main__":
    capture_image()
