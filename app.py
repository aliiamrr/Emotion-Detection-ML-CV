import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image

# Load the trained model and LabelEncoder
svm_model = joblib.load('emotion_classifier.pkl')
le = joblib.load('label_encoder.pkl')

# Preprocessing function (Same steps as during training)
def preprocess_image(image):
    # Check if the image has 3 or 4 channels (RGB or RGBA)
    if len(image.shape) == 3 and image.shape[-1] in [3, 4]:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # If the image is already grayscale, use it as is
        gray = image

    # Resize to 96x96
    resized = cv2.resize(gray, (96, 96))
    # Apply histogram equalization
    equalized = cv2.equalizeHist(resized)
    # Normalize pixel values
    normalized = equalized / 255.0

    return normalized

# HOG feature extraction (Same steps as during training)
def extract_hog_features(image):
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

st.title("Emotion Classifier")

# Added line (Side bar to choose input method (Upload Image or Use Webcam))
input_method = st.sidebar.selectbox("Select Input Method", ("Upload Image", "Use Webcam"))
if input_method == "Upload Image":
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Convert the image to OpenCV format
        image = np.array(image)
        # Handle RGBA images
        if image.shape[-1] == 4:  
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        hog_features = extract_hog_features(preprocessed_image)

        # Predict the emotion
        emotion_label_num = svm_model.predict([hog_features])[0]
        emotion_label_text = le.inverse_transform([emotion_label_num])[0]

        # Display the result
        st.success(f"The detected emotion is: {emotion_label_text}")
        
## ADDITIONAL FUNCTIONALITY: TO USE WEBCAM IN THE APP
elif input_method == "Use Webcam":
    # Start webcam capture
    st.write("Click 'Start Webcam' to begin capturing video.")
    start_webcam = st.button("Start Webcam")

    if start_webcam:
        # OpenCV video capture
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces using Haar Cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                # Extract the face region
                face = gray[y:y+h, x:x+w]

                # Preprocess the face
                preprocessed_face = preprocess_image(face)
                hog_features = extract_hog_features(preprocessed_face)

                # Predict the emotion
                emotion_label_num = svm_model.predict([hog_features])[0]
                emotion_label_text = le.inverse_transform([emotion_label_num])[0]

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display the frame in Streamlit
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        cv2.destroyAllWindows()