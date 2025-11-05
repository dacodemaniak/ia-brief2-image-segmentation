# Image segmentation

Image segmentation takes a picture as input and cut picture into segments with a description of each segement.

## Git repository

[Image Segmentation Repository](https://github.com/dacodemaniak/ia-brief2-image-segmentation.git)

`git clone https://github.com/dacodemaniak/ia-brief2-image-segmentation.git`
`cd ia-brief2-image-segmentation`
`pip install -r requirements.txt`

## Used libs

- streamlit
- pillow
- transformer
- torch
- pytest
- loguru

## Usage

From a terminal (Linux / MacOS) just run :

`./launch.sh`

## CHANGELOG

- [v1.0.0]
    - Segment detection from an image
    - Display both normalized and colorized segments

- [v1.0.1]
    - Using nlpconnect/vit-gpt2-image-captioning to get a text description
    - Update IHM to illustrate each segment
    - Simple TU for segmentation