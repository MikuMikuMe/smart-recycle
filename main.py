Creating an intelligent waste sorting system like "smart-recycle" using image classification and machine learning involves several steps. Below is a complete Python program that outlines the basic framework for such a project. This example uses a pre-trained model from TensorFlow/Keras to classify images into different waste categories. Note that this is a simplified version and assumes access to labeled image data for training the model if needed.

```python
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set path variables
MODEL_PATH = 'path/to/your/model.h5'  # Path to the pre-trained model
CLASS_NAMES = ['plastic', 'paper', 'glass', 'metal', 'organic']  # Example waste categories

def load_trained_model(model_path):
    """
    Load a pre-trained Keras model from a .h5 file.
    """
    try:
        model = load_model(model_path)
        logging.info(f'Model successfully loaded from {model_path}.')
        return model
    except Exception as e:
        logging.error(f'Error loading model: {e}')
        raise ValueError("Could not load the model. Please check the path and try again.")

def preprocess_image(image_path):
    """
    Prepare an image for classification.
    """
    try:
        # Load the image from file
        img = image.load_img(image_path, target_size=(224, 224))
        # Convert the image to an array
        img_array = image.img_to_array(img)
        # Normalize the image
        img_array = img_array / 255.0
        # Expand dimensions to match the model input
        img_array = np.expand_dims(img_array, axis=0)
        logging.info(f'Image at {image_path} successfully preprocessed.')
        return img_array
    except Exception as e:
        logging.error(f'Error preprocessing image: {e}')
        raise ValueError("Could not preprocess the image. Please check the path and try again.")

def classify_waste(model, img_array):
    """
    Classify the waste material in the image.
    """
    try:
        # Get predictions from the model
        predictions = model.predict(img_array)
        # Get index of the highest probability
        predicted_index = np.argmax(predictions)
        # Get the class label
        predicted_class = CLASS_NAMES[predicted_index]
        logging.info(f'Image classified as {predicted_class}.')
        return predicted_class
    except Exception as e:
        logging.error(f'Error classifying image: {e}')
        raise ValueError("Could not classify the image. Please check the model and input data.")

def main():
    # Load the model
    model = load_trained_model(MODEL_PATH)

    # Example image path (to be replaced with actual input logic)
    image_path = 'path/to/your/image.jpg'
    
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Classify the image
    classification_result = classify_waste(model, img_array)

    # Output the result
    print(f"The waste is classified as: {classification_result}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f'An error occurred: {e}')
        print("An error occurred. Please check the logs for more details.")
```

### Explanation:

1. **Logging and Error Handling**: The script uses Python's `logging` module to log messages with different severity levels, allowing for easier debugging. Additionally, error handling is used to gracefully manage issues like loading failures or incorrect image paths.

2. **Model Loading**: The program loads a pre-trained model from disk, assuming the path and format to be specified.

3. **Image Preprocessing**: The input image is loaded and preprocessed (resized, converted to an array, and normalized). It is prepared in a format acceptable to the model.

4. **Classification**: The model predicts the waste category of the input image. It returns the label with the highest probability.

5. **Main Function**: Encapsulates the core logic, including calling load, preprocess, and classify functions. It handles any high-level exceptions by logging an error message and providing feedback.

### Prerequisites:

1. Install the necessary packages using pip: `pip install tensorflow opencv-python`.
2. Ensure that the pre-trained model and the image you want to classify are accessible at the specified file paths.
3. Replace the dummy model path and image path with actual paths in your system.

This framework forms the basis for the smart-recycle system. You can extend it by incorporating real-time image capture, a UI, or connecting it to physical recycling bins for automation.