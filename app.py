import streamlit as st
import torch
import tempfile
import cv2
from PIL import Image
import os

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

# Streamlit UI
st.set_page_config(page_title="🔥 Fire Detection App", layout="centered")
st.title("🔥 Real-Time Fire Detection using YOLOv5")
st.markdown("Created by **Mansi Damahe**")
st.markdown("---")

option = st.radio("Choose Input Type:", ["Image", "Video"])

if option == "Image":
    uploaded_img = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_img is not None:
        st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
        img = Image.open(uploaded_img)
        results = model(img)
        results.render()  # Draw boxes on image

        st.image(results.ims[0], caption="Detected Fire", use_column_width=True)
        st.success("Detection Complete!")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results.render()[0]
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()
        st.success("Video Processing Complete!")

