import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

def get_class_names():
    """Return the list of 38 plant disease class names"""
    return [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

def create_model():
    """Create and return the plant disease classification model"""
    try:
        # Check if a saved model exists
        model_path = 'plant_disease_model.h5'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        
        # If no saved model, create a new one based on the notebook architecture
        # This creates a model structure that can be loaded with pre-trained weights
        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom classification layers
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(38, activation='softmax')  # 38 classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Initialize with random weights if no pre-trained model is available
        # In a real deployment, you would load actual trained weights here
        return model
        
    except Exception as e:
        raise Exception(f"Failed to create model: {str(e)}")

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert PIL image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Normalize pixel values to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise Exception(f"Failed to preprocess image: {str(e)}")

def apply_augmentation(image):
    """Apply data augmentation similar to the training process"""
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Random rotation (up to 90 degrees as in training)
        angle = np.random.uniform(-90, 90)
        height, width = img_array.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        img_array = cv2.warpAffine(img_array, rotation_matrix, (width, height))
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img_array = cv2.flip(img_array, 1)
        
        # Random vertical flip
        if np.random.random() > 0.5:
            img_array = cv2.flip(img_array, 0)
        
        # Convert back to PIL
        return Image.fromarray(img_array)
        
    except Exception as e:
        # Return original image if augmentation fails
        return image
