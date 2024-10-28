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
        # Resize inputs to the target size
        return tf.image.resize(inputs, self.target_size)

    def compute_output_shape(self, input_shape):
        # Define the output shape after resizing
        return (input_shape[0], self.target_size[0], self.target_size[1], input_shape[3])

# Custom layer to scale tensor values by a specified factor
class CustomScaleLayer(Layer):
    def __init__(self, scale_factor=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        # Scale inputs by the factor
        return inputs * self.scale_factor

# Function to preprocess grayscale images by resizing
def preprocess_images(train_data):
    resized_images = tf.image.resize(train_data, [256, 256])  # Resize to 256x256
    return resized_images

# Function to load and prepare grayscale and color images for training
def load_and_prepare_dataset(grayscale_dir, color_dir):
    grayscale_images = []
    color_images = []

    grayscale_filenames = sorted(os.listdir(grayscale_dir))
    color_filenames = sorted(os.listdir(color_dir))

    # Check if the number of grayscale and color images matches
    if len(grayscale_filenames) != len(color_filenames):
        raise ValueError("The number of grayscale and color images do not match.")

    # Load and process each grayscale and color image pair
    for g_filename, c_filename in zip(grayscale_filenames, color_filenames):
        g_img_path = os.path.join(grayscale_dir, g_filename)
        c_img_path = os.path.join(color_dir, c_filename)

        # Load grayscale image, resize, and add single channel dimension
        g_img = cv2.imread(g_img_path, cv2.IMREAD_GRAYSCALE)
        g_img = cv2.resize(g_img, (256, 256))
        g_img = g_img[..., np.newaxis]  # Shape: (256, 256, 1)
        grayscale_images.append(g_img)

        # Load color image, resize, and convert to Lab color space
        c_img = cv2.imread(c_img_path)
        c_img = cv2.resize(c_img, (256, 256))
        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2Lab)
        color_images.append(c_img)

    # Normalize and reshape datasets
    train_data = np.array(grayscale_images, dtype=np.float32) / 255.0  # Grayscale data
    train_labels = np.array(color_images, dtype=np.float32) / 255.0    # Color data
    train_labels = train_labels[:, :, :, 1:3]  # Only keep a and b channels

    return train_data, train_labels

# Function to build the colorization model
def build_colorization_model(input_shape=(256, 256, 1)):
    input_layer = Input(shape=input_shape)

    # Encoder Network (Downsampling)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(input_layer)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)

    # Global Feature Extractor using InceptionResNetV2
    inception = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    inception.trainable = False

    inception_input = Input(shape=(299, 299, 3))  # Input for the Inception model
    inception_output = inception(inception_input)  # Extract features

    # Upsample to match encoder output dimensions
    resized_inception_output = UpSampling2D(size=(4, 4))(inception_output)
    resized_inception_output = Conv2D(256, (1, 1), activation='relu', padding='same')(resized_inception_output)

    # Concatenate encoder and global features
    fusion_layer = concatenate([x, resized_inception_output], axis=-1)

    # Decoder Network (Upsampling)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_layer)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Output layer to produce 2 channels (a and b)
    output_layer = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)

    # Return the model
    return Model(inputs=[input_layer, inception_input], outputs=output_layer), inception

# Function to train the colorization model
def train_model(model, inception_model, train_data, train_labels, epochs=5, batch_size=16):
    if train_data.shape[0] == 0 or train_labels.shape[0] == 0:
        print("No training data available.")
        return

    # Resize grayscale data to match Inception input dimensions
    train_data_resized = tf.image.resize(train_data, [299, 299])
    train_data_resized = np.repeat(train_data_resized, 3, axis=-1)

    # Ensure labels have the correct shape
    train_labels = train_labels.reshape(-1, 256, 256, 2)

    # Compile and train the model
    model.compile(optimizer=Adam(), loss='mse')

    # Train the model
    if len(train_data) >= batch_size:
        model.fit(
            [train_data, train_data_resized],
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1
        )
    else:
        print("Not enough data to perform validation split.")

# Function to colorize a single grayscale image using the model
def colorize_image(model, inception_model, image_path):
    # Load grayscale image and resize for input
    l_channel = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if l_channel is None:
        raise ValueError(f"Image at path '{image_path}' could not be loaded.")
    l_channel_resized = cv2.resize(l_channel, (256, 256)).reshape((256, 256, 1)) / 255.0

    # Prepare inputs for the model
    l_channel_input = np.expand_dims(l_channel_resized, axis=0)
    l_channel_resized_for_inception = np.repeat(cv2.resize(l_channel, (299, 299)).reshape((299, 299, 1)) / 255.0, 3, axis=-1)
    l_channel_resized_for_inception = np.expand_dims(l_channel_resized_for_inception, axis=0)

    # Predict colorization
    ab_channels = model.predict([l_channel_input, l_channel_resized_for_inception])[0]

    # Reassemble colorized image in Lab color space
    l_channel_resized = l_channel_resized.squeeze() * 100.0
    ab_channels = (ab_channels + 1) * 255.0 / 2
    colorized_image = np.zeros((256, 256, 3), dtype=np.uint8)
    colorized_image[:, :, 0] = l_channel_resized
    colorized_image[:, :, 1:3] = ab_channels

    # Convert colorized image to BGR
    return cv2.cvtColor(colorized_image, cv2.COLOR_Lab2BGR)

# Function to save the trained model
def save_trained_model(model, filepath):
    model.save(filepath)
    print(f"Model saved at {filepath}")

# Function to load a saved model with custom layers
def load_saved_model(filepath):
    with tf.keras.utils.custom_object_scope({'ResizeLayer': ResizeLayer, 'CustomScaleLayer': CustomScaleLayer}):
        loaded_model = tf.keras.models.load_model(filepath)
        print("Model loaded successfully.")
    return loaded_model

# Function to calculate Mean Squared Error (MSE) between two images
def calculate_mse(original, generated):
    return np.mean((original - generated) ** 2)

# Function to calculate Peak Signal-to-Noise Ratio (PSNR) between two images
def calculate_psnr(original, generated):
    mse = calculate_mse(original, generated)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


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