from tensorflow.keras.models import load_model

model = load_model("model/lumpy_skin_model.keras")

# Print all layer names
for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name}")
