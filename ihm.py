import streamlit as st
import io
from PIL import Image
from loguru import logger

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
    except Exception as e:
        st.error(f"Something went wrong while image loading {e}")
        logger.error(f"Failed to load : {e}")