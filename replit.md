# Plant Disease Classifier

## Overview

This is a Streamlit-based web application that uses machine learning to classify plant diseases from uploaded images. The application leverages TensorFlow/Keras deep learning models to identify 38 different plant disease classes across various crops including apples, corn, grapes, tomatoes, peppers, and more. Users can upload plant leaf images to receive disease predictions with confidence scores, along with detailed information about symptoms, treatments, and prevention methods.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for rapid prototyping and deployment
- **User Interface**: Single-page application with wide layout and expandable sidebar
- **State Management**: Streamlit session state for model caching and class name persistence
- **Image Handling**: PIL (Python Imaging Library) for image processing and validation

### Backend Architecture
- **ML Framework**: TensorFlow/Keras for deep learning model inference
- **Model Architecture**: Convolutional Neural Network (CNN) trained for multi-class plant disease classification
- **Prediction Pipeline**: Image preprocessing → Model inference → Top-3 predictions with confidence scores
- **Caching Strategy**: Model loaded once per session and cached in session state for performance optimization

### Data Processing
- **Image Preprocessing**: OpenCV and PIL for image resizing, normalization, and format conversion
- **Input Validation**: File size limits (10MB), format validation (JPEG/PNG), minimum dimension requirements (50x50px)
- **Output Formatting**: Confidence scores converted to percentages with visual indicators

### Disease Information System
- **Knowledge Base**: Static dictionary-based disease information database
- **Content Structure**: Each disease entry includes description, symptoms, treatment options, prevention methods, and severity level
- **Coverage**: Comprehensive information for major plant diseases across multiple crop types

### Application Structure
- **Modular Design**: Separated concerns across multiple Python modules
  - `app.py`: Main application logic and Streamlit interface
  - `model_utils.py`: ML model creation and image preprocessing functions
  - `disease_info.py`: Disease information database and retrieval functions
  - `utils.py`: Utility functions for validation and formatting
- **Error Handling**: Comprehensive exception handling for model loading, image processing, and prediction steps

## External Dependencies

### Machine Learning Stack
- **TensorFlow**: Deep learning framework for model inference
- **NumPy**: Numerical computing for array operations and predictions processing

### Image Processing
- **PIL (Pillow)**: Image manipulation and format handling
- **OpenCV (cv2)**: Advanced image preprocessing and computer vision operations

### Web Framework
- **Streamlit**: Web application framework for creating the user interface and handling file uploads

### Development Tools
- Standard Python libraries (os, io) for file system operations and data handling

Note: The application appears to be designed for local deployment or cloud platforms supporting Python web applications. No external APIs, databases, or third-party services are currently integrated, making it a self-contained system.