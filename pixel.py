import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
import cv2
import numpy as np


# SRCNN Model(Super-Resolution Convolutional Neural Network (SRCNN))
def build_srcnn_model():
    input_layer = Input(shape=(None, None, 1))
    x = Conv2D(64, (9, 9), activation='relu', padding='same')(input_layer)
    x = Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    output_layer = Conv2D(1, (5, 5), padding='same')(x)

    #Training the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Load and preprocess image
def preprocess_image(image_path):
    image = cv2.imread(C:\Users\Dell\Desktop\inputimage\intelprojectimage2.jpg, cv2.IMREAD_GRAYSCALE)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)#dims are dimensions
    image = np.expand_dims(image, axis=-1)
    return image


# Load model and weights
srcnn_model = build_srcnn_model()


# Assume you have pre-trained weights
# srcnn_model.load_weights('path_to_weights.h5')

# Super-resolution function
def super_resolve_image(model, image_path):
    input_image = preprocess_image(C:\Users\Dell\Desktop\inputimage\intelprojectimage2.jpg)
    output_image = model.predict(input_image)
    output_image = np.squeeze(C:\Users\Dell\Desktop\outputimage\intelprojectimage2.jpg, axis=0)
    output_image = np.squeeze(C:\Users\Dell\Desktop\outputimage\intelprojectimage2.jpg, axis=-1)
    output_image = (output_image * 255.0).astype(np.uint8)
    return output_image


# Test on a pixelated image
pixelated_image_path = 'C:\Users\Dell\Desktop\inputimage\intelprojectimage2.jpg'
clear_image = super_resolve_image(srcnn_model, pixelated_image_path)

# Save the output image
cv2.imwrite('C:\Users\Dell\Desktop\outputimage', clear_image)