import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Layer
from keras.applications import InceptionResNetV2
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Custom layer to resize tensor using tf.image.resize
class ResizeLayer(Layer):
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.target_size[0], self.target_size[1], input_shape[3])
from tensorflow.keras.layers import Layer
import tensorflow as tf

class CustomScaleLayer(Layer):
    def __init__(self, scale_factor=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return inputs * self.scale_factor

# Image preprocessing function for grayscale images
def preprocess_images(train_data):
    resized_images = tf.image.resize(train_data, [256, 256])  # Adjusted size
    return resized_images

def load_and_prepare_dataset(grayscale_dir, color_dir):
    grayscale_images = []
    color_images = []

    grayscale_filenames = sorted(os.listdir(grayscale_dir))
    color_filenames = sorted(os.listdir(color_dir))

    if len(grayscale_filenames) != len(color_filenames):
        raise ValueError("The number of grayscale and color images do not match.")

    for g_filename, c_filename in zip(grayscale_filenames, color_filenames):
        g_img_path = os.path.join(grayscale_dir, g_filename)
        c_img_path = os.path.join(color_dir, c_filename)

        # Load and resize grayscale images
        g_img = cv2.imread(g_img_path, cv2.IMREAD_GRAYSCALE)
        g_img = cv2.resize(g_img, (256, 256))
        g_img = g_img[..., np.newaxis]  # Ensure it has a single channel shape (256, 256, 1)
        grayscale_images.append(g_img)

        # Load and resize color images
        c_img = cv2.imread(c_img_path)
        c_img = cv2.resize(c_img, (256, 256))
        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2Lab)
        color_images.append(c_img)

    train_data = np.array(grayscale_images, dtype=np.float32) / 255.0  # Shape will be (num_samples, 256, 256, 1)
    train_labels = np.array(color_images, dtype=np.float32) / 255.0
    train_labels = train_labels[:, :, :, 1:3]  # Keep only the a and b channels

    return train_data, train_labels
def build_colorization_model(input_shape=(256, 256, 1)):
    input_layer = Input(shape=input_shape)

    # Encoder Network
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(input_layer)  # (128, 128, 64)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)                     # (128, 128, 128)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)          # (64, 64, 128)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)                     # (64, 64, 256)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)          # (32, 32, 256)

    # Global Feature Extractor using InceptionResNetV2
    inception = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    inception.trainable = False

    inception_input = Input(shape=(299, 299, 3))  # Input for the Inception model
    inception_output = inception(inception_input)  # Forward the input through the Inception model

    # Apply upsampling to match encoder output dimensions
    resized_inception_output = UpSampling2D(size=(4, 4))(inception_output)  # Increase to (32, 32, 256)

    # We want to extract features at a level that can merge properly
    resized_inception_output = Conv2D(256, (1, 1), activation='relu', padding='same')(resized_inception_output)

    # Debugging: Print shapes to confirm dimensions
    print("Shape of Encoder Output (x):", x.shape)  # Should be (32, 32, 256)
    print("Shape of Resized Inception Output:", resized_inception_output.shape)  # Should be (32, 32, 256)

    # Concatenate layers
    fusion_layer = concatenate([x, resized_inception_output], axis=-1)  # Now both should be compatible

    # Decoder Network
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_layer)
    x = UpSampling2D((2, 2))(x)  # Now should be (64, 64, 128)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # Now should be (128, 128, 64)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)  # Now should be (256, 256, 32)

    # Final output layer should produce 2 channels (a and b)
    output_layer = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)  # Output layer

    # Return the model
    return Model(inputs=[input_layer, inception_input], outputs=output_layer), inception

def train_model(model, inception_model, train_data, train_labels, epochs=5, batch_size=16):
    if train_data.shape[0] == 0 or train_labels.shape[0] == 0:
        print("No training data available. Please check the dataset.")
        return

    # Debugging: Print the shape of train_data before preprocessing
    print("Shape of train_data before preprocessing:", train_data.shape)

    # Prepare features for training using the Inception model
    # Resize and repeat grayscale to 3 channels for Inception input
    train_data_resized = tf.image.resize(train_data, [299, 299])  # Resize to 299x299
    train_data_resized = np.repeat(train_data_resized, 3, axis=-1)  # Repeat to 3 channels

    # Ensure labels are in the correct shape for training
    train_labels = train_labels.reshape(-1, 256, 256, 2)  # Ensure 2 channels for a and b
    print("train_data shape:", train_data.shape)            # Should be (N, 256, 256, 1)
    print("train_data_resized shape:", train_data_resized.shape)  # Should be (N, 299, 299, 3)
    print("train_labels shape:", train_labels.shape)        # Should be (N, 256, 256, 2)

    # Compile and train the model
    model.compile(optimizer=Adam(), loss='mse')

    # Ensure enough data for validation split
    if len(train_data) >= 16:  # Make sure there are enough samples for batch size
        model.fit(
            [train_data, train_data_resized],  # Use resized data for Inception input
            train_labels,                       # Should have shape (N, 256, 256, 2)
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1  # Adjust based on your dataset size
        )
    else:
        print("Not enough data to perform validation split.")
def colorize_image(model, inception_model, image_path):
    # Load and preprocess the image
    l_channel = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if l_channel is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded. Check the path.")

    # Resize the grayscale image to 256x256 for L channel
    l_channel_resized = cv2.resize(l_channel, (256, 256))
    l_channel_resized = l_channel_resized.reshape((256, 256, 1)) / 255.0  # Normalize L channel

    # Prepare the input for the model (for L channel)
    l_channel_input = np.expand_dims(l_channel_resized, axis=0)  # Shape (1, 256, 256, 1)

    # Prepare the input for the Inception model
    l_channel_resized_for_inception = cv2.resize(l_channel, (299, 299))  # Resize for Inception model
    l_channel_resized_for_inception = l_channel_resized_for_inception.reshape((299, 299, 1)) / 255.0
    l_channel_resized_for_inception = np.repeat(l_channel_resized_for_inception, 3, axis=-1)  # Repeat to 3 channels
    l_channel_resized_for_inception = np.expand_dims(l_channel_resized_for_inception, axis=0)  # Shape (1, 299, 299, 3)

    # Generate colorization output
    ab_channels = model.predict([l_channel_input, l_channel_resized_for_inception])[0]  # Get a and b channels

    # Prepare the colorized output in Lab color space
    l_channel_resized = l_channel_resized.squeeze() * 100.0  # Scale back to [0, 100]
    ab_channels = (ab_channels + 1) * 255.0 / 2  # Scale a and b channels back to [0, 255]

    colorized_image = np.zeros((256, 256, 3), dtype=np.uint8)
    colorized_image[:, :, 0] = l_channel_resized  # L channel
    colorized_image[:, :, 1:3] = ab_channels  # a and b channels

    # Convert back to BGR color space
    colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_Lab2BGR)

    return colorized_image
def save_trained_model(model, filepath):
    model.save(filepath)
    print(f"Model saved at {filepath}")

# Load the model for prediction
def load_saved_model(filepath):
    # Register the custom layer here
    loaded_model = tf.keras.models.load_model(filepath, custom_objects={'ResizeLayer': ResizeLayer})
    print("Model loaded successfully.")
    return loaded_model

def calculate_mse(original, generated):
    """Calculate Mean Squared Error between original and generated images."""
    return np.mean((original - generated) ** 2)

def calculate_psnr(original, generated):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = calculate_mse(original, generated)
    if mse == 0:
        return float('inf')  # No noise, images are identical
    max_pixel = 255.0  # Assuming pixel values are in the range [0, 255]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Example usage (ensure you have your model properly defined)
grayscale_dir = r'D:\Repos\CV\data\train_black'
color_dir = r'D:\Repos\CV\data\train_color'
train_data, train_labels = load_and_prepare_dataset(grayscale_dir, color_dir)


# Ensure train_data is in the right shape
train_data_resized = preprocess_images(train_data)

# Ensure the labels are also in the correct shape for training
train_labels_resized = train_labels.reshape(-1, 256, 256, 2)  # Change to 2 channels for a and b

# Build the model
model, inception_model = build_colorization_model()

# Train the model
train_model(model, inception_model, train_data_resized, train_labels_resized, epochs=4, batch_size=16)
model_save_path = r'D:\Repos\CV\colorization_model.h5'


save_trained_model(model, model_save_path)

from tensorflow.keras.utils import custom_object_scope

from tensorflow.keras.utils import custom_object_scope

def load_saved_model(filepath):
    # Register both custom layers here
    with custom_object_scope({'ResizeLayer': ResizeLayer, 'CustomScaleLayer': CustomScaleLayer}):
        loaded_model = tf.keras.models.load_model(r'D:\Repos\CV\colorization_model.h5', custom_objects={'ResizeLayer': ResizeLayer, 'CustomScaleLayer': CustomScaleLayer})
        print("Model loaded successfully.")
    return loaded_model

image_path = r'D:\Repos\CV\data\train_black\image0001.jpg'
original_color_image = cv2.imread(image_path)
original_color_image = cv2.resize(original_color_image, (256, 256))
original_color_image = cv2.cvtColor(original_color_image, cv2.COLOR_BGR2Lab)  # Convert to Lab color space
original_ab_channels = original_color_image[:, :, 1:3]
colorized_result = colorize_image(model, inception_model, image_path)

mse = calculate_mse(original_ab_channels, (colorized_result[:, :, 1:3] / 255.0))
psnr = calculate_psnr(original_ab_channels, (colorized_result[:, :, 1:3] / 255.0))

# Print the results
print(f'Mean Squared Error (MSE): {mse}')
print(f'Peak Signal-to-Noise Ratio (PSNR): {psnr} dB')

cv2.imwrite('colorized_image.jpg', (colorized_result * 255).astype(np.uint8))