import torch
import numpy as np
from PIL import Image
import io
from loguru import logger
import streamlit as st
from transformers import DetrFeatureExtractor, DetrForSegmentation
from transformers.models.detr.feature_extraction_detr import rgb_to_id

logger.add("logs/process.log", rotation="500 MB", level="INFO")

@st.cache_resource
def load_models():
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
    model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

    return feature_extractor, model


# First load models
feature_extractor, model = load_models()

def pre_process_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_array = np.array(image)

    if len(image_array.shape) == 2: # Gray level
        image_array = np.stack([image_array] * 3, axis=1)
    elif image_array.shape[2] == 4: # RGBA picture
        image_array = image_array[:, :, :3]
    
    return Image.fromarray(image_array)

def image_segmentation(image):
    try: 
        '''
            Pre treat image to convert if necessary
        '''
        processed_image = pre_process_image(image)

        # Prepare image for the model
        inputs = feature_extractor(images=processed_image, return_tensors="pt")

        # Disable gradients for inference
        with torch.no_grad():
            outputs = model(**inputs)


        processed_sizes = torch.as_tensor(inputs["pixel_values"].shape[-2:]).unsqueeze(0)
        result = feature_extractor.post_process_panoptic(outputs, processed_sizes)[0]

        # Segmentation is stored in special-format png
        panoptic_seg = Image.open(io.BytesIO(result["png_string"]))
        panoptic_seg = np.array(panoptic_seg, dtype=np.uint8)
        
        # retrieve the ids corresponding to each mask and returns it
        segmenation_map = rgb_to_id(panoptic_seg)

        return segmenation_map
    except Exception as e:
        logger.error(f"Error in image_segmentation {e}")
        raise e
    
def normalize_segmentation(segments):
    """Segment normalisation for diplaying"""
    if np.max(segments) > 0:
        normalized = (segments / np.max(segments) * 255).astype(np.uint8)
    else:
        normalized = segments.astype(np.uint8)
    return normalized


def colorize_segmentation(segments):
    """Colorize map"""
    # random colors for each segment
    unique_ids = np.unique(segments)
    color_map = {}
    
    for seg_id in unique_ids:
        if seg_id == 0:  # Black background
            color_map[seg_id] = [0, 0, 0]
        else:
            # Random color for segmetns
            color_map[seg_id] = np.random.randint(0, 255, 3)
    
    # Apply color card
    colored = np.zeros((*segments.shape, 3), dtype=np.uint8)
    for seg_id, color in color_map.items():
        colored[segments == seg_id] = color
    
    return colored