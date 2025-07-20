import cv2
from paddleocr import PaddleOCR
import re
from fuzzywuzzy import fuzz
import numpy as np

def load_image_from_file(image_path):
    # Load an image from the file system
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    return image

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
    # Define target words and regex patterns
    target_words = {"example", "keyword", "test", "hello"}  # Modify as needed
    regex_patterns = [r"\btest\d+\b", r"\bhello world\b"]  # Example regex patterns
    
    # # Preprocess the image
    # processed_image = preprocess_image(image)
    
    # Perform OCR using PaddleOCR
    ocr = PaddleOCR()
    result = ocr.ocr(image, cls=True)
    
    # Extract detected text
    extracted_text = " ".join([word_info[1][0].lower() for line in result for word_info in line])
    
    print("Extracted Text:")
    print(extracted_text)
    
    # Check for approximate word matches using fuzzy matching
    matched_words = set()
    for target in target_words:
        similarity = fuzz.ratio(extracted_text, target)
        if similarity >= 80:  # Adjust threshold as needed
            matched_words.add(target)
    
    # Check for regex pattern matches
    matched_patterns = set()
    for pattern in regex_patterns:
        if re.search(pattern, extracted_text):
            matched_patterns.add(pattern)
    
    if matched_words:
        print(f"Matched Words (Fuzzy Matching): {matched_words}")
    if matched_patterns:
        print(f"Matched Patterns (Regex): {matched_patterns}")
    if not matched_words and not matched_patterns:
        print("No target words or patterns found.")

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    image = load_image_from_file(image_path)
    if image is not None:
        extract_text(image)