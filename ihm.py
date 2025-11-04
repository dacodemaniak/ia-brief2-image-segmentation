import streamlit as st
import io
from PIL import Image
from loguru import logger

import numpy as np
from process import image_segmentation, normalize_segmentation, colorize_segmentation

logger.add("logs/ihm.log", rotation="500 MB", level="INFO")

st.title("Image segmentation")

'''
    Upload file
'''
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats JPG, PNG"
)

if uploaded_file is not None:
    st.success("File was successfully loaded")
    logger.info(f"An image was loaded : {uploaded_file.name}")

    # Use PIL to load and display image
    try: 
        image = Image.open(uploaded_file)
        st.image(image, caption="Loaded image", use_container_width=True)

        # Segmentation process
        with st.spinner("Processing image segmentation..."):
            segments = image_segmentation(image=image)

        st.success("Segmentation completed successfully")

        st.subheader("Debug information")
        st.write(f"Segmentation map shape: {segments.shape}")
        st.write(f"Min: {np.min(segments)}, Max: {np.max(segments)}")
        st.write(f"Number of unique : {len(np.unique(segments))}")

        st.subheader("Segmentation Map")
        col1, col2 = st.columns(2)

        with col1:
            st.write("** Normalized segmentation")
            normalized_seg = normalize_segmentation(segments)
            st.image(normalized_seg, caption="Normalized", use_container_width=True, clamp=True)

        with col2:
            st.write("** Colorized segmentation")
            colorized_seg = colorize_segmentation(segments)
            st.image(colorized_seg, caption="Colorized", use_container_width=True, clamp=True)

    except Exception as e:
        st.error(f"Something went wrong while image loading {e}")
        logger.error(f"Failed to load : {e}")