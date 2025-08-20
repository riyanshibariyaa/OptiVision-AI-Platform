# REPLACE your TextExtraction.py with this lazy loading version:

import cv2
import fitz
import docx
import os
import numpy as np
from PIL import Image

# LAZY LOADING - Don't initialize PaddleOCR at import time
ocr = None

def get_ocr_instance():
    """Get OCR instance with lazy loading"""
    global ocr
    if ocr is None:
        print("Loading PaddleOCR on demand...")
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        print("PaddleOCR loaded successfully")
    return ocr

# Function to extract text from an image file
def extract_text_from_image(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Unable to read the image file."
        
        # Lazy load OCR only when actually needed
        ocr_instance = get_ocr_instance()
        
        # Convert image to grayscale for better OCR performance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image_path = "processed_image.jpg"
        cv2.imwrite(processed_image_path, gray)
        
        # Perform OCR using PaddleOCR
        result = ocr_instance.ocr(processed_image_path, cls=True)
        
        extracted_text = "\n".join([line[1][0] for res in result for line in res])
        return extracted_text if extracted_text else "No readable text found in the image."
    except Exception as e:
        return f"Error processing image: {str(e)}"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        return f"Error reading PDF: {e}"
    return text.strip() if text.strip() else "No readable text found in PDF."

# Function to extract text from a Word (.docx) file
def extract_text_from_docx(docx_path):
    text = ""
    try:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        return f"Error reading DOCX: {e}"
    return text.strip() if text.strip() else "No readable text found in DOCX."

# Function to extract text from a plain text file
def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read().strip()
            return text if text else "No readable text found in TXT."
    except Exception as e:
        return f"Error reading TXT: {e}"

# Function to determine file type and extract text accordingly
def extract_text(file_path):
    if not os.path.exists(file_path):
        return "Error: File does not exist!"

    file_ext = file_path.lower().split('.')[-1]

    if file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        return extract_text_from_image(file_path)
    elif file_ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_ext == 'docx':
        return extract_text_from_docx(file_path)
    elif file_ext == 'txt':
        return extract_text_from_txt(file_path)
    else:
        return f"Unsupported file type: {file_ext}"
