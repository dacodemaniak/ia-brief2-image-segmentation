import streamlit as st
import io
from PIL import Image
from loguru import logger

import numpy as np
from process import image_segmentation, normalize_segmentation, colorize_segmentation

logger.add("logs/ihm.log", rotation="500 MB", level="INFO")

st.set_page_config(page_title="Image segmentation and captioning", layout="wide")

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
            result = image_segmentation(image=image)

        st.success("Segmentation completed successfully")

        st.subheader("Segmentation Map")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Segmentation map")
            colorized_seg = colorize_segmentation(result['segmentation_map'])
            st.image(colorized_seg, caption="Colorized", use_container_width=True, clamp=True)

        with col2:
            st.subheader(f"Segments Found: {len(result['descriptions'])}")
            st.write(f"Segmentation shape: {result['segmentation_map'].shape}")

        # Descriptions display
        st.subheader("Segment descriptions")

        for i, desc_data in enumerate(result['descriptions']):
            with st.expander(f"Segment {desc_data['segment_id']} - {desc_data['description']}", expanded=i==0):
                col_img, col_desc = st.columns([1, 2])

                with col_img:
                    st.image(desc_data['image'], caption=f"Segment {desc_data['segment_id']}", use_column_width=True)

                with col_desc:
                    st.write(f"Description : {desc_data['description']}")
                    st.write(f"Bounding Box : {desc_data['bbox']}")
                    st.write(f"Segment ID : {desc_data['segment_id']}")                                

    except Exception as e:
        st.error(f"Something went wrong while image loading {e}")
        logger.error(f"Failed to load : {e}")