"""
MNIST CNN Digit Recognizer - Interactive Streamlit Web Application
===================================================================
This application provides an interactive interface for training and using
a Convolutional Neural Network to recognize handwritten digits (0-9) from
the MNIST dataset.

Features:
- Automatic model training and caching
- Interactive digit prediction from uploaded images
- Visualization of random predictions from test set
- Training history and performance metrics

Author: AI Assistant
Date: 2024
"""

# ===================================================================
# SECTION 1: IMPORTS
# ===================================================================

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import io

# Set page configuration - must be the first Streamlit command
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ===================================================================
# SECTION 2: CONSTANTS AND CONFIGURATION
# ===================================================================

MODEL_PATH = "mnist_cnn_model.h5"
IMG_SIZE = 28
NUM_CLASSES = 10
BATCH_SIZE = 128
EPOCHS = 10
VALIDATION_SPLIT = 0.1

# ===================================================================
# SECTION 3: DATA LOADING AND PREPROCESSING
# ===================================================================

@st.cache_data
def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset.
    
    Returns:
        tuple: (x_train, y_train, x_test, y_test) - Preprocessed datasets
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to include channel dimension
    x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    return x_train, y_train, x_test, y_test

# ===================================================================
# SECTION 4: MODEL DEFINITION
# ===================================================================

def create_cnn_model():
    """
    Create and compile a Convolutional Neural Network for MNIST digit classification.
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=(IMG_SIZE, IMG_SIZE, 1), name='conv1'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),
        
        # Third Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', name='conv3'),
        
        # Flattening and Dense Layers
        layers.Flatten(name='flatten'),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        
        # Output Layer
        layers.Dense(NUM_CLASSES, activation='softmax', name='output')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ===================================================================
# SECTION 5: MODEL TRAINING AND LOADING
# ===================================================================

@st.cache_resource
def load_or_train_model():
    """
    Load a pre-trained model if available, otherwise train a new one.
    
    Returns:
        tuple: (model, history, test_accuracy) - Trained model and training metrics
    """
    # Load preprocessed data
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Check if saved model exists
    if os.path.exists(MODEL_PATH):
        st.info(f"📂 Loading existing model from '{MODEL_PATH}'...")
        model = keras.models.load_model(MODEL_PATH)
        
        # Evaluate the loaded model
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        st.success(f"✅ Model loaded successfully! Test Accuracy: {test_accuracy*100:.2f}%")
        
        return model, None, test_accuracy
    
    else:
        st.warning(f"⚠️ No saved model found. Training new model...")
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        epoch_metrics = st.empty()
        
        # Create model
        model = create_cnn_model()
        
        # Custom callback to update Streamlit UI during training
        class StreamlitCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / EPOCHS
                progress_bar.progress(progress)
                status_text.text(f"Training Progress: Epoch {epoch + 1}/{EPOCHS}")
                
                # Display metrics
                metrics_text = f"""
                **Epoch {epoch + 1} Metrics:**
                - Training Accuracy: {logs['accuracy']*100:.2f}%
                - Validation Accuracy: {logs['val_accuracy']*100:.2f}%
                - Training Loss: {logs['loss']:.4f}
                - Validation Loss: {logs['val_loss']:.4f}
                """
                epoch_metrics.markdown(metrics_text)
        
        # Train the model
        history = model.fit(
            x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT,
            verbose=0,  # Suppress default output
            callbacks=[StreamlitCallback()]
        )
        
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        
        # Save the trained model
        model.save(MODEL_PATH)
        
        progress_bar.progress(1.0)
        status_text.text("Training Complete!")
        
        st.success(f"✅ Model trained and saved! Test Accuracy: {test_accuracy*100:.2f}%")
        
        return model, history, test_accuracy

# ===================================================================
# SECTION 6: IMAGE PREPROCESSING FOR UPLOADED FILES
# ===================================================================

def preprocess_uploaded_image(uploaded_file):
    """
    Preprocess an uploaded image for model prediction.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        tuple: (processed_image, original_image) - Processed array and PIL Image
    """
    try:
        # Open image using PIL
        image = Image.open(uploaded_file)
        
        # Convert to grayscale
        image_gray = image.convert('L')
        
        # Resize to 28x28
        image_resized = image_gray.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image_resized)
        
        # Normalize pixel values to [0, 1]
        image_normalized = image_array.astype('float32') / 255.0
        
        # Reshape for model input (1, 28, 28, 1)
        image_processed = image_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        
        return image_processed, image_resized
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

# ===================================================================
# SECTION 7: PREDICTION FUNCTIONS
# ===================================================================

def predict_digit(model, image_array):
    """
    Predict the digit from a preprocessed image array.
    
    Args:
        model: Trained Keras model
        image_array: Preprocessed image array
        
    Returns:
        tuple: (predicted_label, confidence, all_probabilities)
    """
    # Get prediction probabilities
    predictions = model.predict(image_array, verbose=0)
    
    # Get predicted class
    predicted_label = np.argmax(predictions[0])
    
    # Get confidence
    confidence = predictions[0][predicted_label] * 100
    
    return predicted_label, confidence, predictions[0]

# ===================================================================
# SECTION 8: VISUALIZATION FUNCTIONS
# ===================================================================

def visualize_random_predictions(model, x_test, y_test, num_samples=5):
    """
    Visualize predictions on random test samples.
    
    Args:
        model: Trained Keras model
        x_test: Test images
        y_test: Test labels
        num_samples: Number of samples to visualize
        
    Returns:
        matplotlib.figure.Figure: Figure with predictions
    """
    # Select random samples
    random_indices = random.sample(range(len(x_test)), num_samples)
    
    # Get predictions
    predictions = model.predict(x_test[random_indices], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    fig.suptitle('CNN Predictions on MNIST Test Set', fontsize=16, fontweight='bold')
    
    # Plot each sample
    for i, (idx, ax) in enumerate(zip(random_indices, axes)):
        image = x_test[idx].reshape(IMG_SIZE, IMG_SIZE)
        true_label = y_test[idx]
        pred_label = predicted_labels[i]
        confidence = predictions[i][pred_label] * 100
        
        # Display image
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        
        # Set color based on correctness
        color = 'green' if pred_label == true_label else 'red'
        
        # Create title
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    return fig

def plot_probability_distribution(probabilities):
    """
    Plot the probability distribution for all digit classes.
    
    Args:
        probabilities: Array of probabilities for each class
        
    Returns:
        matplotlib.figure.Figure: Bar chart of probabilities
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    classes = list(range(NUM_CLASSES))
    colors = ['green' if p == max(probabilities) else 'skyblue' for p in probabilities]
    
    ax.bar(classes, probabilities * 100, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Digit Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(classes)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_training_history(history):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Keras History object from model training
        
    Returns:
        matplotlib.figure.Figure: Training history plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', marker='o')
    ax2.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

# ===================================================================
# SECTION 9: MAIN STREAMLIT APPLICATION
# ===================================================================

def main():
    """
    Main function to run the Streamlit application.
    """
    
    # ===================================================================
    # HEADER SECTION
    # ===================================================================
    
    st.title("🔢 MNIST Handwritten Digit Recognizer")
    st.markdown("""
    Welcome to the **Interactive MNIST CNN Classifier**! This application uses a 
    Convolutional Neural Network to recognize handwritten digits (0-9) with high accuracy.
    """)
    
    st.markdown("---")
    
    # ===================================================================
    # SIDEBAR CONFIGURATION
    # ===================================================================
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown(f"""
        **Model Details:**
        - Architecture: CNN (3 Conv Layers)
        - Input Size: {IMG_SIZE}×{IMG_SIZE} grayscale
        - Classes: {NUM_CLASSES} (digits 0-9)
        - Training Epochs: {EPOCHS}
        - Batch Size: {BATCH_SIZE}
        """)
        
        st.markdown("---")
        
        st.header("📖 Instructions")
        st.markdown("""
        1. Wait for model to load/train
        2. Upload a digit image (28×28 recommended)
        3. View prediction results
        4. Explore random predictions from test set
        """)
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.markdown("""
        This app demonstrates deep learning for image classification 
        using the famous MNIST dataset.
        
        **Technologies:**
        - TensorFlow/Keras
        - Streamlit
        - NumPy/Matplotlib
        """)
    
    # ===================================================================
    # MODEL LOADING/TRAINING SECTION
    # ===================================================================
    
    st.header("🤖 Model Status")
    
    with st.spinner("Loading model... Please wait."):
        model, history, test_accuracy = load_or_train_model()
    
    # Display model metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Test Accuracy",
            value=f"{test_accuracy*100:.2f}%",
            delta="Above 95% ✓" if test_accuracy > 0.95 else "Below 95% ✗"
        )
    
    with col2:
        st.metric(
            label="🎯 Target Accuracy",
            value="95.00%"
        )
    
    with col3:
        model_status = "✅ Excellent" if test_accuracy > 0.95 else "⚠️ Needs Improvement"
        st.metric(
            label="🏆 Model Status",
            value=model_status
        )
    
    # Show training history if model was just trained
    if history is not None:
        st.markdown("---")
        st.subheader("📈 Training History")
        fig_history = plot_training_history(history)
        st.pyplot(fig_history)
        plt.close()
    
    st.markdown("---")
    
    # ===================================================================
    # INTERACTIVE PREDICTION SECTION
    # ===================================================================
    
    st.header("🖼️ Upload Your Digit Image")
    st.markdown("""
    Upload an image of a handwritten digit (0-9). For best results, use a 28×28 pixel 
    grayscale image with a white digit on a black background (similar to MNIST format).
    """)
    
    uploaded_file = st.file_uploader(
        "Choose an image file (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a handwritten digit"
    )
    
    if uploaded_file is not None:
        st.markdown("---")
        st.subheader("🔍 Prediction Results")
        
        # Process the uploaded image
        processed_image, display_image = preprocess_uploaded_image(uploaded_file)
        
        if processed_image is not None:
            # Create columns for display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Uploaded Image:**")
                st.image(display_image, caption="Preprocessed Image (28×28)", width=200)
            
            with col2:
                # Make prediction
                with st.spinner("Analyzing image..."):
                    predicted_digit, confidence, probabilities = predict_digit(model, processed_image)
                
                # Display prediction
                st.markdown("### Prediction:")
                st.markdown(f"## **The digit is: {predicted_digit}**")
                st.markdown(f"**Confidence: {confidence:.2f}%**")
                
                # Show confidence level indicator
                if confidence > 90:
                    st.success("🎯 Very High Confidence!")
                elif confidence > 70:
                    st.info("👍 Good Confidence")
                else:
                    st.warning("⚠️ Low Confidence - Image may be unclear")
            
            # Display probability distribution
            st.markdown("---")
            st.subheader("📊 Confidence Distribution Across All Digits")
            fig_probs = plot_probability_distribution(probabilities)
            st.pyplot(fig_probs)
            plt.close()
            
            # Show detailed probabilities
            with st.expander("📋 View Detailed Probabilities"):
                prob_df_data = {
                    "Digit": list(range(NUM_CLASSES)),
                    "Probability (%)": [f"{p*100:.2f}%" for p in probabilities]
                }
                st.table(prob_df_data)
    
    else:
        st.info("👆 Please upload an image to get started!")
    
    st.markdown("---")
    
    # ===================================================================
    # RANDOM PREDICTIONS VISUALIZATION SECTION
    # ===================================================================
    
    st.header("🎨 Explore Test Set Predictions")
    st.markdown("""
    Click the button below to view predictions on random samples from the MNIST test dataset.
    Green titles indicate correct predictions, red titles indicate errors.
    """)
    
    if st.button("🎲 Show Random Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            # Load test data
            _, _, x_test, y_test = load_and_preprocess_data()
            
            # Generate visualization
            fig_random = visualize_random_predictions(model, x_test, y_test, num_samples=5)
            st.pyplot(fig_random)
            plt.close()
            
            st.success("✅ Predictions generated successfully!")
    
    st.markdown("---")
    
    # ===================================================================
    # FOOTER SECTION
    # ===================================================================
    
    st.header("📚 Model Architecture")
    
    with st.expander("🔍 View Detailed Model Architecture"):
        # Capture model summary
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = "\n".join(stringlist)
        
        st.code(model_summary, language="text")
    
    st.markdown("---")
    
    # Final footer
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🎉 <strong>MNIST CNN Digit Recognizer</strong> 🎉</p>
        <p>Built with TensorFlow, Keras, and Streamlit</p>
        <p>Achieving 95%+ accuracy on handwritten digit recognition</p>
    </div>
    """, unsafe_allow_html=True)

# ===================================================================
# SECTION 10: APPLICATION ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    main()