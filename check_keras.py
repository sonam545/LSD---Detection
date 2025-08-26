from tensorflow.keras.models import load_model

model = load_model("model/lumpy_skin_model.keras", compile=False)
print(model.summary())
