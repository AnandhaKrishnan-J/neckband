import easyocr
import cv2
from fuzzywuzzy import fuzz
import regex as re

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the language

# Load the image
image_path = "C:/Users/anand/Downloads/100_rupee.jpg"  # Replace with your image path
image = cv2.imread(image_path)
target_words = {"bank", "reserve", "india"}
# Perform text detection
results = reader.readtext(image_path)
matched_words = set()
serial_code_found = False
pattern = r"(\d{2}[A-Z]){e<=1}(\s+){e<=2}(\d{6}){e<=2}" 
compiled_pattern = re.compile(pattern)

# Draw bounding boxes and display detected text
for (bbox, text, prob) in results:
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))
    
    # Draw bounding box
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    
    # Put text
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 0, 255), 2)
    
    print(f'Detected text: {text}, Confidence: {prob:.2f}')
    if prob > 0.65:
        print("in check :",text)
        for target in target_words:
            text = text.lower()
            similarity = fuzz.ratio(text, target)
            if similarity >= 50:  # Adjust threshold as needed
                matched_words.add(target)
        if (re.search(compiled_pattern, text)):
            print("pattern maych :",text)
            serial_code_found = True
            break
print(matched_words)
if matched_words and serial_code_found:
    print("Note valid")
# Show the image with detections
cv2.imshow('Text Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
