import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from model_utils import create_model, preprocess_image, get_class_names
from disease_info import get_disease_info
from utils import validate_image, format_confidence

# Page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = get_class_names()

def load_model():
    """Load the pre-trained model"""
    try:
        if st.session_state.model is None:
            with st.spinner("Loading AI model... This may take a moment."):
                st.session_state.model = create_model()
        return st.session_state.model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def classify_image(image, model):
    """Classify the uploaded image"""
    try:
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        results = []
        for idx in top_indices:
            confidence = float(predictions[0][idx])
            class_name = st.session_state.class_names[idx]
            results.append({
                'disease': class_name,
                'confidence': confidence,
                'percentage': confidence * 100
            })
        
        return results
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return None

def main():
    # Header
    st.title("🌱 Plant Disease Classifier")
    st.markdown("### AI-powered disease detection for healthier crops")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 How to Use")
        st.markdown("""
        1. **Upload** a clear photo of the affected plant leaf
        2. **Wait** for AI analysis (few seconds)
        3. **Review** the disease classification results
        4. **Follow** the treatment recommendations
        """)
        
        st.header("📸 Photo Guidelines")
        st.markdown("""
        - Use good lighting
        - Focus on the affected leaf
        - Avoid blurry images
        - Supported formats: JPG, JPEG, PNG
        - Max file size: 10MB
        """)
        
        st.header("🔬 Supported Diseases")
        st.markdown("This AI model can identify **38 different** plant diseases across various crops including:")
        st.markdown("- Tomato diseases")
        st.markdown("- Potato diseases") 
        st.markdown("- Corn diseases")
        st.markdown("- Apple diseases")
        st.markdown("- And many more...")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Plant Image")
        uploaded_file = st.file_uploader(
            "Choose an image of the affected plant leaf",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of the plant leaf showing disease symptoms"
        )
        
        if uploaded_file is not None:
            # Validate image
            if not validate_image(uploaded_file):
                st.error("Invalid image file. Please upload a valid JPG, JPEG, or PNG image.")
                return
            
            # Display uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.info(f"📊 Image size: {image.size[0]}x{image.size[1]} pixels")
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                return
    
    with col2:
        st.header("🔍 Analysis Results")
        
        if uploaded_file is not None:
            # Load model
            model = load_model()
            
            if model is not None:
                # Classify button
                if st.button("🔬 Analyze Plant Disease", type="primary", use_container_width=True):
                    with st.spinner("Analyzing image... Please wait"):
                        results = classify_image(image, model)
                        
                        if results:
                            st.success("✅ Analysis Complete!")
                            
                            # Display top prediction
                            top_result = results[0]
                            confidence_color = "🟢" if top_result['confidence'] > 0.7 else "🟡" if top_result['confidence'] > 0.5 else "🔴"
                            
                            st.markdown(f"### {confidence_color} Primary Diagnosis")
                            st.markdown(f"**Disease:** {top_result['disease']}")
                            st.markdown(f"**Confidence:** {format_confidence(top_result['confidence'])}")
                            
                            # Confidence bar
                            st.progress(top_result['confidence'])
                            
                            # Alternative predictions
                            if len(results) > 1:
                                st.markdown("### 📊 Alternative Possibilities")
                                for i, result in enumerate(results[1:], 2):
                                    with st.expander(f"{i}. {result['disease']} ({format_confidence(result['confidence'])})"):
                                        disease_info = get_disease_info(result['disease'])
                                        if disease_info:
                                            st.markdown(f"**Description:** {disease_info['description']}")
                            
                            # Detailed information for top prediction
                            st.markdown("---")
                            disease_info = get_disease_info(top_result['disease'])
                            
                            if disease_info:
                                st.markdown("### 📖 Disease Information")
                                
                                # Create tabs for organized information
                                tab1, tab2, tab3 = st.tabs(["📝 Description", "🩺 Treatment", "🛡️ Prevention"])
                                
                                with tab1:
                                    st.markdown(disease_info['description'])
                                    if disease_info.get('symptoms'):
                                        st.markdown("**Common Symptoms:**")
                                        for symptom in disease_info['symptoms']:
                                            st.markdown(f"• {symptom}")
                                
                                with tab2:
                                    st.markdown("**Recommended Treatment:**")
                                    for treatment in disease_info['treatment']:
                                        st.markdown(f"• {treatment}")
                                    
                                    if disease_info.get('severity') == 'high':
                                        st.warning("⚠️ This is a serious disease. Consider consulting an agricultural expert.")
                                
                                with tab3:
                                    st.markdown("**Prevention Measures:**")
                                    for prevention in disease_info['prevention']:
                                        st.markdown(f"• {prevention}")
                            
                            # Disclaimer
                            st.markdown("---")
                            st.info("ℹ️ **Disclaimer:** This AI tool provides guidance only. For serious infections or uncertainty, consult with local agricultural experts or extension services.")
            
            else:
                st.error("❌ Model failed to load. Please refresh the page and try again.")
        
        else:
            st.info("👆 Upload an image to start the analysis")
            
            # Sample images section
            st.markdown("### 🖼️ Need sample images?")
            st.markdown("Try taking photos of:")
            st.markdown("- Leaves with spots or discoloration")
            st.markdown("- Wilted or yellowing foliage") 
            st.markdown("- Plants with visible fungal growth")
            st.markdown("- Any unusual plant symptoms")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🌾 Built for farmers, by farmers. Helping grow healthier crops through AI technology.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
