import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Layer
from keras.applications import InceptionResNetV2
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(original, generated):
    """Calculate Mean Squared Error (MSE) between the original and generated images.
    
    Args:
        original (np.ndarray): Original image (a and b channels only).
        generated (np.ndarray): Generated image (a and b channels only).
        
    Returns:
        float: Mean Squared Error between original and generated images.
    """
    return np.mean((original - generated) ** 2)

# Function to calculate Peak Signal-to-Noise Ratio (PSNR)
def calculate_psnr(original, generated):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between original and generated images.
    
    Args:
        original (np.ndarray): Original image (a and b channels only).
        generated (np.ndarray): Generated image (a and b channels only).
        
    Returns:
        float: PSNR value in dB.
    """
    mse = calculate_mse(original, generated)
    if mse == 0:
        return float('inf')  # If MSE is 0, images are identical, and PSNR is infinite
    max_pixel = 255.0  # Assuming pixel values are in range [0, 255]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Load custom layers for a pre-trained model
from tensorflow.keras.utils import custom_object_scope

# Function to load a saved colorization model
def load_saved_model(filepath):
    """Load a saved model with custom layers (ResizeLayer and CustomScaleLayer).
    
    Args:
        filepath (str): Path to the saved model file.
        
    Returns:
        tf.keras.Model: Loaded Keras model with custom layers.
    """
    # Register both custom layers in custom_object_scope
    with custom_object_scope({'ResizeLayer': ResizeLayer, 'CustomScaleLayer': CustomScaleLayer}):
        loaded_model = tf.keras.models.load_model(filepath)
        print("Model loaded successfully.")
    return loaded_model

# Path to grayscale image for colorization
image_path = r'D:\Repos\CV\data\train_black\image0001.jpg'

# Load the original color image for comparison
original_color_image = cv2.imread(image_path)
original_color_image = cv2.resize(original_color_image, (256, 256))
original_color_image = cv2.cvtColor(original_color_image, cv2.COLOR_BGR2Lab)  # Convert to Lab color space
original_ab_channels = original_color_image[:, :, 1:3]  # Extract the a and b channels for comparison

# Colorize the grayscale image using the loaded model
colorized_result = colorize_image(model, inception_model, image_path)

# Calculate MSE and PSNR between original and colorized images
mse = calculate_mse(original_ab_channels, (colorized_result[:, :, 1:3] / 255.0))
psnr = calculate_psnr(original_ab_channels, (colorized_result[:, :, 1:3] / 255.0))

# Print the results of MSE and PSNR calculations
print(f'Mean Squared Error (MSE): {mse}')
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr} dB')

# Save the colorized image as a JPEG file
cv2.imwrite('colorized_image.jpg', (colorized_result * 255).astype(np.uint8))
