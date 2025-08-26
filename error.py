from tensorflow.keras.models import load_model

try:
    model = load_model("model/lumpy_skin_model.keras", compile=False)
    print("Model loaded successfully ✅")
except Exception as e:
    print("❌ Error loading model:", e)
