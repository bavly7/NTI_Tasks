import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Set page config
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Load your trained model from the weights directory
@st.cache_resource
def load_flower_model():
    try:
        # Update the path to your model
        model_path = 'Fower_Tasks/Weights/flower_classifier.keras'
        model = load_model(model_path)
        st.success(f"✅ Model loaded successfully from: {model_path}")
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: {model_path}")
        st.info("Please make sure 'weights/flower_classifier.keras' exists in the project directory.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Preprocess the image to match model input (32x32)
def preprocess_image(img):
    # Resize to 32x32
    img = img.resize((32, 32))
    # Convert to array
    img_array = np.array(img)
    
    # If image has alpha channel, remove it
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    # If image is grayscale, convert to RGB
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Main app
def main():
    st.title("🌸 Flower Classification App")
    st.write("Upload an image of a flower to identify its type!")
    
    # Define the 5 flower classes (adjust based on your training order!)
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    
    # Load model
    model = load_flower_model()
    
    if model is None:
        st.warning("Please make sure 'weights/flower_classifier.keras' exists and is accessible.")
        
        # Show current directory structure for debugging
        with st.expander("📁 Current directory contents"):
            if os.path.exists('.'):
                for root, dirs, files in os.walk('.'):
                    level = root.replace('.', '').count(os.sep)
                    indent = ' ' * 2 * level
                    st.write(f"{indent}📂 {os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        st.write(f"{subindent}📄 {file}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a flower image...", 
        type=["jpg", "jpeg", "png", "bmp"],
        help="Supported formats: JPG, JPEG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        try:
            # Open and display the image
            img = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(img, use_column_width=True)
                
                # Show resized preview
                resized_img = img.resize((32, 32))
                st.image(resized_img, caption="Resized to 32x32 (model input)", width=100)
                
                # Show image info
                st.caption(f"Original size: {img.size[0]}x{img.size[1]} pixels")
                st.caption(f"Format: {img.format}")
            
            with col2:
                st.subheader("Classification Results")
                
                # Preprocess image
                processed_img = preprocess_image(img)
                
                # Make prediction
                predictions = model.predict(processed_img, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                predicted_class = class_names[predicted_class_idx]
                confidence = predictions[0][predicted_class_idx]
                
                # Display prediction
                st.success(f"**Prediction:** {predicted_class.title()}")
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Show all probabilities
                st.subheader("All Probabilities:")
                for i, (cls, prob) in enumerate(zip(class_names, predictions[0])):
                    # Create a colored bar for each class
                    color = "green" if i == predicted_class_idx else "blue"
                    st.write(f"**{cls.title()}:** {prob:.2%}")
                    st.progress(float(prob))
                    
        except Exception as e:
            st.error(f"Error processing image: {e}")

    # Add some instructions
    with st.expander("ℹ️ How to use this app"):
        st.write("""
        1. **Upload an image** of a flower using the file uploader above
        2. The app will resize it to 32x32 pixels (as required by the model)
        3. The AI model will analyze the image and predict the flower type
        4. You'll see the top prediction along with confidence scores for all 5 flower types
        
        **Supported flower types:**
        - Daisy 🌼
        - Dandelion 🌸  
        - Rose 🌹
        - Sunflower 🌻
        - Tulip 🌷
        
        **Note:** For best results, use clear, centered images of single flowers.
        """)

    # Add footer
    st.markdown("---")
    st.caption("Built with TensorFlow, Keras, and Streamlit | Model input size: 32x32 pixels")

if __name__ == "__main__":
    main()