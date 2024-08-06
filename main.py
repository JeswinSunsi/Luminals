"""
File: main.py
Authors: Jeswin Sunsi & Afnaan TK
Date: 03-08-2024
Description: PDF PII Redactor - Pair Programming Hackathon 2024
Made with Coffee and Code by Team Luminals
"""

import fitz
import spacy
import cv2
import numpy as np
import os, io, re
from PIL import Image, ImageDraw, ImageFilter
import pytesseract
from presidio_analyzer import AnalyzerEngine
from wand.image import Image as wi
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# TODO: Implement safe CORS
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Regex to reduce NER workload
EMAIL_REG = r"([\w\.\d]+\@\s?[\w\d]+\.[\w\d]+)"
NUMBER_REG = r"((\+*)((0[ -]*)*|((91 )*))((\d{12})+|(\d{10})+))|\d{5}([- ]*)\d{6}"
PLATE_REG = r"^[A-Z]{2}[ -][0-9]{1,2}(?: [A-Z])?(?: [A-Z]*)? [0-9]{4}$"
ZIP_REG = r"^\d{6}$"
YEAR_REG = r"^\d{4}$"

color_mask = "BLACK"
isUsingHaar = False
isUsingPresidio = False

analyzer = AnalyzerEngine()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
nlp = spacy.load("en_core_web_sm")

output_pdf_path = "redacted.pdf"

def get_sensitive_data(lines):
    global isUsingPresidio
    if not(isUsingPresidio): # Use base NER (Spacy)
        for line in lines:
            for word in line.split():
                if re.search(EMAIL_REG, word, re.IGNORECASE): # Mail redaction
                    search = re.search(EMAIL_REG, word, re.IGNORECASE)
                    yield search.group(1)
                if "@" in word:
                    yield word
                if re.search(NUMBER_REG, word, re.IGNORECASE): # Number redaction
                    search = re.search(NUMBER_REG, word, re.IGNORECASE)
                    yield search.group(1)
                if re.search(PLATE_REG, word, re.IGNORECASE): # Vehicle plate redaction
                    search = re.search(PLATE_REG, word, re.IGNORECASE)
                    try:
                        yield search.group(1)
                    except:
                        pass
                if re.search(ZIP_REG, word, re.IGNORECASE): # Zip redaction
                    search = re.search(ZIP_REG, word, re.IGNORECASE)
                    try:
                        yield search.group(1)
                    except:
                        pass
                if re.search(YEAR_REG, word, re.IGNORECASE): # Year redaction
                    search = re.search(YEAR_REG, word, re.IGNORECASE)
                    try:
                        yield search.group(1)
                    except:
                        pass

        doc = nlp(str(lines)) # SpacyNER load
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "GPE"}:
                text = ent.text.replace("of", "")
                yield text
    
    else: # Use Presidio NER
        for line in lines:
            results = analyzer.analyze(text=str(line), language='en')
            for item in results:
                numbers = re.findall(r'\d+', str(item))
                numbers = [int(num) for num in numbers]
                if len(str(line[numbers[0]:numbers[1]])) == 2 and "al" not in str(line[numbers[0]:numbers[1]]):
                    pass
                else:
                    yield str(line[numbers[0]:numbers[1]])

def redact_text_with_pymupdf(doc):
    for page in doc:
        page.wrap_contents()
        raw_sensitive = list(get_sensitive_data(page.get_text("text").split('\n')))
        with open("redacted_data.txt", "w") as handle:
            for word in raw_sensitive:
                handle.write(word+"\n")
        sensitive = []
        for word in raw_sensitive:
            temp_list = word.split()
            for temp in temp_list:
                if temp not in ["al","the", "-"] and len(temp) > 2: # Ignore edge cases
                    sensitive.append(temp)
        for data in sensitive:
            raw_areas = page.search_for(data)
            for area in raw_areas:
                extracted_text = page.get_text("text", clip=area).strip()
                if extracted_text == data:
                    # TODO: Implement switch case
                    if color_mask == "BLACK":
                        page.add_redact_annot(area, fill=(0, 0, 0))
                    elif color_mask == "WHITE":
                        page.add_redact_annot(area, fill=(1.0, 1.0, 1.0))
                    elif color_mask == "RED":
                        page.add_redact_annot(area, fill=(1.0, 0.0, 0.0))
                    elif color_mask == "ORANGE":
                        page.add_redact_annot(area, fill=(1.0, 0.64, 0.0))
                    elif color_mask == "GREEN":
                        page.add_redact_annot(area, fill=(0.0, 1.0, 0.0))
                    elif color_mask == "BLUE":
                        page.add_redact_annot(area, fill=(0.0, 0.0, 1.0))
        page.apply_redactions()
    return doc

def blur_text_in_image(image, boxes):
    draw = ImageDraw.Draw(image)
    for (x, y, w, h) in boxes:
        # TODO: Implement switch case
        if color_mask == "BLACK":
            draw.rectangle([x, y, x + w, y + h], fill="black")
        elif color_mask == "WHITE":
            draw.rectangle([x, y, x + w, y + h], fill="white")
        elif color_mask == "RED":
            draw.rectangle([x, y, x + w, y + h], fill="red")
        elif color_mask == "GREEN":
            draw.rectangle([x, y, x + w, y + h], fill="green")
        elif color_mask == "BLUE":
            draw.rectangle([x, y, x + w, y + h], fill="blue")
    return image

def process_image_with_tesseract(pdf_path):
    pdf = wi(filename=pdf_path, resolution=300)
    pdf_img = pdf.convert('jpeg')
    img_blobs = []

    for img in pdf_img.sequence:
        page = wi(image=img)
        img_blobs.append(page.make_blob('jpeg'))

    processed_images = []
    for img_blob in img_blobs:
        im = Image.open(io.BytesIO(img_blob))
        text = pytesseract.image_to_string(im, lang='eng')
        lines = text.split('\n')
        raw_sensitive = list(get_sensitive_data(lines))
        # Formatting sensitive items into word/word
        sensitive = []
        for word in raw_sensitive:
            temp_list = word.split()
            for temp in temp_list:
                # Edge cases
                if temp not in ['the', "al", "-"] and len(temp) > 2:
                    sensitive.append(temp)
        d = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
        boxes_to_blur = []

        for i in range(len(d['text'])):
            word = d['text'][i]
            if word in sensitive:
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                boxes_to_blur.append((x, y, w, h))

        im = blur_text_in_image(im, boxes_to_blur)
        processed_images.append(im)

    if processed_images:
        processed_images[0].save(
            output_pdf_path, save_all=True, append_images=processed_images[1:], resolution=100.0, quality=95
        )

def process_image(pdf_path, output_pdf_path): # Haar classifier for Face-Blur
    pdf = wi(filename=pdf_path, resolution=300)
    pdf_img = pdf.convert('jpeg')
    face_classifier_frontal = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    processed_images = []
    for i, img in enumerate(pdf_img.sequence): #TODO: remove extra dep
        page = wi(image=img)
        image_blob = page.make_blob('jpeg')
        image_np = np.frombuffer(image_blob, np.uint8)
        image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        faces_frontal = face_classifier_frontal.detectMultiScale(image_cv2)
        for (x, y, w, h) in faces_frontal:
            roi = image_cv2[y:y+h, x:x+w]
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            image_cv2[y:y+h, x:x+w] = blurred_roi
        processed_image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        processed_images.append(processed_image_pil)
    if processed_images:
        processed_images[0].save(
            output_pdf_path, save_all=True, append_images=processed_images[1:], resolution=100.0, quality=95
        )

# Home Route
@app.get("/")
async def default_home():
    return {"info": "development API for Luminals"}

@app.get("/v1/color/{mask_color}")
async def changeMaskColor(mask_color: str):
    global color_mask 
    color_mask = mask_color
    return {"Color Mask": color_mask}

@app.get("/v1/presidio/")
async def togglePresidio():
    global isUsingPresidio 
    isUsingPresidio = not(isUsingPresidio)
    return {"isUsingPresidio": isUsingPresidio}

@app.get("/v1/haar/")
async def toggleHaar():
    global isUsingHaar 
    isUsingHaar = not(isUsingHaar)
    return {"isUsingHaar": isUsingHaar}

@app.get("/v1/start/{pdf_path}")
async def startPoint(pdf_path: str):
    doc = fitz.open(pdf_path)
    first_page = doc[0]
    text = first_page.get_text("text")
    if text.strip():
        # Case 1: PyMuPDF
        doc = redact_text_with_pymupdf(doc)
        doc.save(output_pdf_path)
        print("200: Redaction complete C1")
    else:
        # Case 2: Tesseract OCR
        print("100: Switching Engines")
        process_image_with_tesseract(pdf_path)
        print("200: Redaction complete C2")
    global isUsingHaar
    if isUsingHaar: # Face Blur
        process_image('redacted.pdf', 'redacted.pdf')
    return {"200": "Ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
