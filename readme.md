# Image Colorization Using Deep Learning

This project aims to colorize grayscale images using a deep learning model that combines a Convolutional Neural Network (CNN) for feature extraction and InceptionResNetV2 for enhanced colorization. The model is trained on grayscale images and their color counterparts to learn mappings between grayscale pixel values and color values in the Lab color space.

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Results](#results)
* [Model Evaluation](#model-evaluation)
* [Acknowledgements](#acknowledgements)

---

## Features

* **Custom Layers** : Uses custom TensorFlow layers (`ResizeLayer`, `CustomScaleLayer`) to handle dynamic resizing and scaling within the model.
* **InceptionResNetV2 Integration** : Leverages pretrained InceptionResNetV2 for robust feature extraction.
* **Evaluation Metrics** : Supports MSE (Mean Squared Error) and PSNR (Peak Signal-to-Noise Ratio) for evaluating colorization quality.

## Installation

  **Clone the repository** :
```bash

   git clone https://github.com/duvarakeshss/colorizeAI
   cd colorizeAI
```
   **Install dependencies** :
   ```bash
   pip install -r requirements.txt
   ```
   Key dependencies:

* `TensorFlow`
* `Keras`
* `OpenCV`
* `NumPy`

**Set up directory structure** :
Organize your images as:
   ```bash

   ├── data
   │   ├── train_black    # Grayscale training images
   │   └── train_color    # Color training images
   ```
## Model Evaluation

To measure colorization quality:

mse = calculate_mse(original_ab_channels, generated_ab_channels)
psnr = calculate_psnr(original_ab_channels, generated_ab_channels)
print(f'MSE: {mse}, PSNR: {psnr}')


## Results

| Metric | Value    |
| ------ | -------- |
| MSE    | x.xx     |
| PSNR   | xx.xx dB |

## Acknowledgements

This project is based on recent advances in deep learning for image processing, particularly the use of pretrained networks such as InceptionResNetV2. The custom layers and image processing logic are built using Keras and TensorFlow.
