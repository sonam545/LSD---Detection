from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from generate_report import generate_pdf_report
import numpy as np

# Data generator (same as training)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = datagen.flow_from_directory('clean_data', subset='validation', target_size=(224, 224), batch_size=32, class_mode='binary', shuffle=False)

# Load the trained model
model = load_model('model/lumpy_skin_model.keras')

# Predict
y_true = val_gen.classes
y_prob = model.predict(val_gen).ravel()
y_pred = (y_prob > 0.5).astype(int)

generate_pdf_report(
    output_path="reports/LSD_Report.pdf",
    prediction=predicted_class,
    confidence=confidence_score,
    gradcam_path="gradcam_output.jpg",
    farmer_id="Farmer01",
    cattle_id="Cow12"
)

# Evaluation
print("Classification Report:\n", classification_report(y_true, y_pred, target_names=['Healthy', 'Infected']))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
