import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import os
from gtts import gTTS
import cv2
from PIL import Image
import time
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model
from generate_report import create_pdf_report

st.set_page_config(page_title="Lumpy Skin Disease Detection", layout="wide")

# ---------------------------
# Load Model (Cached)
# ---------------------------
@st.cache_resource
def load_my_model():
    return load_model("../model/lumpy_skin_model.keras", compile=False)

model = load_my_model()
class_names = ["Healthy", "Infected"]
last_conv_layer_name = "conv5_block3_3_conv"  # make sure this exists in your model
model.summary()

# ---------------------------
# Translations
# ---------------------------
languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn"
}
translations = {
    "Healthy": {
        "en": "Cow is Healthy",
        "hi": "गाय स्वस्थ है",
        "ta": "மாடு ஆரோக்கியமாக உள்ளது",
        "te": "ఎద్దు ఆరోగ్యంగా ఉంది",
        "bn": "গরুটি সুস্থ আছে"
    },
    "Infected": {
        "en": "Cow is Infected",
        "hi": "गाय संक्रमित है",
        "ta": "மாடு பாதிக்கப்பட்டுள்ளது",
        "te": "ఎద్దు సంక్రమించింది",
        "bn": "গরুটি সংক্রামিত"
    }
}

# ---------------------------
# Grad-CAM Function
# ---------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    img_array = tf.convert_to_tensor(img_array)

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[0][pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



# ---------------------------
# Prediction Function
# ---------------------------
def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    confidence = float(preds[0])
    result = "Infected" if confidence > 0.5 else "Healthy"
    return result, round(confidence if confidence > 0.5 else 1 - confidence, 2), img_array

# ---------------------------
# Voice Feedback
# ---------------------------
def speak_result(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        tts.save(temp_audio.name)
        temp_audio_path = temp_audio.name

    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio_path)
    pygame.mixer.music.play()

    # Allow playback for a few seconds without freezing Streamlit
    time.sleep(3)

    pygame.mixer.quit()
    os.remove(temp_audio_path)

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("🐄 Lumpy Skin Disease Detection")
st.write("Upload or capture cow image for prediction.")

lang_choice = st.selectbox("Choose Language for Voice Feedback", list(languages.keys()))
option = st.radio("Choose input type", ["Upload Image", "Use Webcam"])
# ---------------------------
# Upload Image Flow
# ---------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload cow image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            result, confidence, img_array = predict(tmp.name)

            # Show prediction
            st.success(f"Prediction: {result} ({confidence * 100:.2f}%)")

            # Speak translated result
            speak_result(translations[result][languages[lang_choice]], languages[lang_choice])

            # Grad-CAM heatmap
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

            import matplotlib.cm as cm
            img_cv = cv2.imread(tmp.name)
            img_cv = cv2.resize(img_cv, (224, 224))
            heatmap = np.uint8(255 * heatmap)

            jet = cm.get_cmap("jet")
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]
            jet_heatmap = cv2.resize(jet_heatmap, (img_cv.shape[1], img_cv.shape[0]))
            jet_heatmap = np.uint8(jet_heatmap * 255)

            superimposed_img = cv2.addWeighted(img_cv, 0.6, jet_heatmap, 0.4, 0)
            st.image(superimposed_img, caption="Grad-CAM Heatmap", use_column_width=True)

            gradcam_path = "reports/gradcam_temp.jpg"
            os.makedirs("reports", exist_ok=True)
            cv2.imwrite(gradcam_path, superimposed_img)

            pdf_path = create_pdf_report(
            prediction=result,
            confidence=confidence,
            gradcam_path=gradcam_path,
            farmer_name=st.text_input("Farmer Name (optional)"),
            location=st.text_input("Location (optional)"),
            breed=st.text_input("Cow Breed (optional)"),
            age=st.text_input("Cow Age (optional)")
        )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf"
                )
# ---------------------------
# Webcam Flow
# ---------------------------
elif option == "Use Webcam":
    st.warning("Click 'Start' to activate webcam.")

    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.result = ""
            self.conf = 0.0
            self.frame_count = 0

        def transform(self, frame):
            self.frame_count += 1
            img = frame.to_ndarray(format="bgr24")
            if self.frame_count % 30 == 0:  # ~1 sec
                img_resized = cv2.resize(img, (224, 224)) / 255.0
                img_input = np.expand_dims(img_resized, axis=0)
                preds = model.predict(img_input)[0]
                conf = float(preds[0])
                result = "Infected" if conf > 0.5 else "Healthy"
                self.result = result
                self.conf = round(conf if conf > 0.5 else 1 - conf, 2)

            cv2.putText(img, f"{self.result} ({self.conf*100:.1f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return img

    webrtc_streamer(key="live", video_processor_factory=VideoProcessor)
