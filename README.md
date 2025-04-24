#streamlit run app.py
#use this to run

import streamlit as st
import sqlite3
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np
from datetime import datetime

# Initialize BLIP model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Initialize database
def init_db():
    conn = sqlite3.connect('caption_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS captions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  image_path TEXT,
                  caption TEXT,
                  created_at TIMESTAMP)''')
    conn.commit()
    conn.close()

# Save caption to database
def save_caption(image_path, caption):
    conn = sqlite3.connect('caption_database.db')
    c = conn.cursor()
    c.execute("INSERT INTO captions (image_path, caption, created_at) VALUES (?, ?, ?)",
              (image_path, caption, datetime.now()))
    conn.commit()
    conn.close()

# Generate caption for image
def generate_caption(image, processor, model):
    # Convert to PIL Image if it's a numpy array (from webcam)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Preprocess image
    inputs = processor(image, return_tensors="pt")
    
    # Generate caption
    with torch.no_grad():
        output = model.generate(**inputs)
    
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def main():
    st.set_page_config(page_title="AI Image Caption Generator", page_icon="📷")
    
    # Initialize model and database
    processor, model = load_model()
    init_db()
    
    st.title("AI-Powered Image Caption Generator")
    st.write("Upload an image or capture from webcam to generate a descriptive caption.")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Webcam Capture"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Generate Caption"):
                caption = generate_caption(image, processor, model)
                st.success(f"Generated Caption: {caption}")
                
                # Save to database
                save_path = os.path.join("images", uploaded_file.name)
                os.makedirs("images", exist_ok=True)
                image.save(save_path)
                save_caption(save_path, caption)
    
    with tab2:
        st.write("Click the button below to capture an image from your webcam.")
        if st.button("Open Webcam"):
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open webcam")
                return
            
            ret, frame = cap.read()
            if ret:
                st.image(frame, channels="BGR", caption="Captured Image", use_column_width=True)
                
                if st.button("Generate Caption for Webcam Image"):
                    caption = generate_caption(frame, processor, model)
                    st.success(f"Generated Caption: {caption}")
                    
                    # Save to database
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join("images", f"webcam_{timestamp}.jpg")
                    os.makedirs("images", exist_ok=True)
                    cv2.imwrite(save_path, frame)
                    save_caption(save_path, caption)
            
            cap.release()
    
    # Display caption history
    st.subheader("Caption History")
    conn = sqlite3.connect('caption_database.db')
    history = conn.execute("SELECT image_path, caption, created_at FROM captions ORDER BY created_at DESC").fetchall()
    conn.close()
    
    if history:
        for img_path, caption, created_at in history:
            with st.expander(f"{caption} - {created_at}"):
                try:
                    st.image(img_path, caption=caption, width=300)
                except:
                    st.warning("Image not found")
    else:
        st.info("No caption history yet")

if __name__ == "__main__":
    main()
