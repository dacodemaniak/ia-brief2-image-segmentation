import pytest
import sys
import os
from PIL import Image, ImageDraw
import numpy as np

# Sets path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from process import image_segmentation, colorize_segmentation

@pytest.fixture
def test_image():
    """ Fixture for generating test image """
    width, height = 400, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Add some simple shapes
    draw.rectangle([50, 50, 150, 150], fill='red', outline='black')
    draw.ellipse([200, 100, 300, 200], fill='blue', outline='black')
    draw.polygon([(350, 50), (320, 150), (380, 150)], fill='green')

    return image

@pytest.fixture
def fake_segmentation_map():
    """ Segment map faked """
    return np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 2, 2]
    ])

def test_basic_segmentation(test_image):
    print("\n Basic segmentation test")

    # Save test image
    test_image.save("test_image.png")

    result = image_segmentation(test_image)

    # Base assertions
    assert 'segmentation_map' in result
    assert 'descriptions' in result
    assert 'original_image' in result
    