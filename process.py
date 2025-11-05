import torch
import numpy as np
from PIL import Image
import io
from loguru import logger
import streamlit as st
from transformers import DetrFeatureExtractor, DetrForSegmentation, pipeline
from transformers.models.detr.feature_extraction_detr import rgb_to_id

logger.add("logs/process.log", rotation="500 MB", level="INFO")

# Sets some globals
_feature_extractor = None
_model = None
_image_captioner = None


# Once then cached avoiding lags
@st.cache_resource
def load_models():
    # Get backs globals
    global _feature_extractor, _model, _image_captioner

    if _feature_extractor is None or _model is None: 
        _feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")
        _model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")

    if _image_captioner is None:
        _image_captioner = pipeline(
            "image-to-text",
            model="nlpconnect/vit-gpt2-image-captioning",
            device=0 if torch.cuda.is_available() else -1 # Use GPU is available
        )
    return _feature_extractor, _model, _image_captioner

def extract_segment_images(original_image, segments):
    '''
        Extract individual images of each segment
    '''

    # Ensure original image and segments are same size, resize if not
    original_array = np.array(original_image)

    if original_array.shape[:2] != segments.shape:
        # Resize segment
        seg_resized = Image.fromarray(segments.astype(np.uint8))
        seg_resized = seg_resized.resize(
            (original_array.shape[1], original_array.shape[0]),
            Image.NEAREST
        )
        segments = np.array(seg_resized)
        
    unique_segments = np.unique(segments)
    segment_images = []

    for seg_id in unique_segments:
        if seg_id == 0: # Simply ignore background
            continue
        # Create a mask for segment
        mask = (segments == seg_id)

        # Convert original image to array
        original_array = np.array(original_image)

        # Build an image with only the segment (transparent or black background)
        segment_array = original_array.copy()
        segment_array[~mask] = 0 # Each pixel out of segment passed to 0

        # Convert to PIL image
        segment_img = Image.fromarray(segment_array)

        # Find segment limit to crop
        rows, cols = np.where(mask)
        if len(rows) > 0 and len(cols) > 0:
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)

            # Crop image to get only segment
            cropped_segment = segment_img.crop((min_col, min_row, max_col, max_row))
            segment_images.append({
                'segment_id': seg_id,
                'image': cropped_segment,
                'bbox': (min_col, min_row, max_col, max_row)
            })
    return segment_images
    
def generate_segment_description(segment_images, max_length=30):
    '''
        Generate descriptions for each segment
    '''
    _, _, captioner = load_models()

    descriptions = []

    for segment_data in segment_images:
        try:
            # Segment description builder
            results = captioner(
                segment_data['image'],
                max_new_tokens = max_length,
                num_beams = 3 # To improve description quality
            )

            description = results[0]['generated_text']

            descriptions.append({
                'segment_id': segment_data['segment_id'],
                'description': description,
                'bbox': segment_data['bbox'],
                'image': segment_data['image']
            })
        except Exception as e:
            logger.error(f"Error generating description for segment {segment_data['segment_id']}")
            descriptions.append({
                'segment_id': segment_data['segment_id'],
                'description': 'No description available for this segment',
                'bbox': segment_data['bbox'],
                'image': segment_data['image']
            })
    return descriptions

# First load models
feature_extractor, model, _ = load_models()

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
        feature_extractor, model, _ = load_models()

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
        segmentation_map = rgb_to_id(panoptic_seg)

        # Extract images from segments
        segment_images = extract_segment_images(image, segmentation_map)

        # Generate description
        descriptions = generate_segment_description(segment_images)

        return {
            'segmentation_map': segmentation_map,
            'descriptions': descriptions,
            'original_image': image
        }
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