import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from skimage.feature import hog
import joblib
from PIL import Image
import os

'''
Note: ZAMN Temporarily removed the local option for webcam input and will only use the Streamlit Cloud option.
'''

# Load the trained model and LabelEncoder
svm_model = joblib.load('emotion_classifier.pkl')
le = joblib.load('label_encoder.pkl')

# # Checking if the app is running locally or on Streamlit Cloud (NEW)
# # Let the user manually toggle cloud mode from the sidebar
# is_cloud = st.sidebar.checkbox("Running on Streamlit Cloud?", value=True)

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

# Title
st.title("Emotion Classifier")

# About Section:
st.markdown("""
This application uses a machine learning model to detect emotions from images or webcam input.  
It demonstrates skills in **computer vision** and **machine learning**.
""")

# How it works section:
st.header("How It Works")
st.markdown("""
1. **Image Upload or Webcam Input**: Users can upload an image or use their webcam from the sidebar options.
2. **Preprocessing**: The image is converted to grayscale, resized, and normalized.
3. **Feature Extraction**: HOG (Histogram of Oriented Gradients) features are extracted.
4. **Emotion Prediction**: A pre-trained SVM model predicts the emotion.
5. **Result Display**: The detected emotion is displayed on the screen.
""")

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
    st.write("Click 'Start' to begin capturing video.")
    # if is_cloud:
    class EmotionDetectionTransformer(VideoTransformerBase):
                def transform(self, frame):
                    img = frame.to_ndarray(format="bgr24")

                    # Detect faces
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        face = gray[y:y+h, x:x+w]
                        preprocessed_face = preprocess_image(face)
                        hog_features = extract_hog_features(preprocessed_face)

                        # Predict emotion
                        emotion_label_num = svm_model.predict([hog_features])[0]
                        emotion_label_text = le.inverse_transform([emotion_label_num])[0]

                        # Draw rectangle and label
                        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(img, emotion_label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                    return img
#     webrtc_streamer(
#     key="emotion-detection",
#     video_transformer_factory=EmotionDetectionTransformer,
#     rtc_configuration={
#         "iceServers": [
#             {"urls": "stun:stun.l.google.com:19302"}
#         ]
#     }
# )

    # ENVIRONMENT VARIABLES FOR TWILIO CONFIGURATION
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

    webrtc_streamer(
        key="emotion-detection",
        video_transformer_factory=EmotionDetectionTransformer,
        rtc_configuration = {
            "iceServers": [
                {"urls": "stun:global.stun.twilio.com:3478"},
                {
                    "urls": "turn:global.turn.twilio.com:3478",
                    "username": TWILIO_ACCOUNT_SID,
                    "credential": TWILIO_AUTH_TOKEN
                },
                {
                    "urls": "turn:global.turn.twilio.com:443",
                    "username": TWILIO_ACCOUNT_SID,
                    "credential": TWILIO_AUTH_TOKEN
                }
            ]
        }
) 
    # else: 
    #     st.write("Click 'Start Webcam' to begin capturing video.")
    #     # Create placeholders for buttons
    #     start_button_placeholder = st.empty()
    #     stop_button_placeholder = st.empty()

    #     # Display the Start Webcam button
    #     start_webcam = start_button_placeholder.button("Start Webcam")

    #     if start_webcam:
    #         # Hide the Start Webcam button
    #         start_button_placeholder.empty()

    #         # OpenCV video capture
    #         cap = cv2.VideoCapture(0)
    #         stframe = st.empty()

    #         # Display the Stop Webcam button
    #         stop_webcam = stop_button_placeholder.button("Stop Webcam")

    #         while True:
    #             # Close webcam if 'Stop Webcam' button is pressed
    #             if stop_webcam:
    #                 st.write("Webcam stopped.")
    #                 cap.release()
    #                 cv2.destroyAllWindows()
    #                 # Hide the Stop Webcam button
    #                 stop_button_placeholder.empty()  
    #                 break

    #             ret, frame = cap.read()
    #             if not ret:
    #                 st.error("Failed to capture video")
    #                 break

    #             # Convert frame to grayscale
    #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #             # Detect faces using Haar Cascade
    #             face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #             faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #             for (x, y, w, h) in faces:
    #                 # Extract the face region
    #                 face = gray[y:y+h, x:x+w]

    #                 # Preprocess the face
    #                 preprocessed_face = preprocess_image(face)
    #                 hog_features = extract_hog_features(preprocessed_face)

    #                 # Predict the emotion
    #                 emotion_label_num = svm_model.predict([hog_features])[0]
    #                 emotion_label_text = le.inverse_transform([emotion_label_num])[0]

    #                 # Draw rectangle and label
    #                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #                 cv2.putText(frame, emotion_label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    #             # Display the frame in Streamlit
    #             stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    #         # Release resources when the loop ends
    #         cap.release()
    #         cv2.destroyAllWindows()