from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import os
import uuid

def generate_unique_ids():
    cow_id = "COW-" + str(uuid.uuid4())[:8]  # Short UUID
    farmer_id = "FR-" + datetime.now().strftime("%Y%m%d%H%M%S")  # Timestamp
    report_id = "RPT-" + datetime.now().strftime("%Y%m%d%H%M%S")
    return farmer_id, cow_id, report_id

def create_pdf_report(prediction, confidence, gradcam_path=None, save_dir="reports",
                      farmer_name=None, location=None, breed=None, age=None):
    os.makedirs(save_dir, exist_ok=True)

    farmer_id, cow_id, report_id = generate_unique_ids()
    filename = os.path.join(save_dir, f"{report_id}.pdf")

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 50, "🐄 Lumpy Skin Disease Detection Report")

    # Report Info
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 90, f"Report ID: {report_id}")
    c.drawString(50, height - 110, f"Date & Time: {datetime.now().strftime('%d-%b-%Y %I:%M %p')}")

    # Farmer Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 150, "Farmer Information")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 170, f"Farmer ID: {farmer_id}")
    c.drawString(70, height - 190, f"Name: {farmer_name if farmer_name else 'N/A'}")
    c.drawString(70, height - 210, f"Location: {location if location else 'N/A'}")

    # Cow Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 250, "Cow Information")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 270, f"Cow ID: {cow_id}")
    c.drawString(70, height - 290, f"Breed: {breed if breed else 'N/A'}")
    c.drawString(70, height - 310, f"Age: {age if age else 'N/A'} years")

    # Detection Results
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 350, "Detection Results")
    c.setFont("Helvetica", 12)
    c.drawString(70, height - 370, f"Status: {prediction.upper()}")
    c.drawString(70, height - 390, f"Confidence: {confidence * 100:.2f}%")

    # Grad-CAM Image
    if gradcam_path and os.path.exists(gradcam_path):
        c.drawImage(gradcam_path, 70, height - 590, width=300, height=180)

    # Health Summary
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 620, "Health Summary")
    c.setFont("Helvetica", 12)
    if prediction.lower() == "healthy":
        c.drawString(70, height - 640, "No visible signs of Lumpy Skin Disease detected.")
    else:
        c.drawString(70, height - 640, "Lesions detected consistent with Lumpy Skin Disease.")

    # Treatment Advisory
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 680, "Treatment / Advisory")
    c.setFont("Helvetica", 12)
    if prediction.lower() == "infected":
        advisory = [
            "1. Isolate the cow immediately from the herd.",
            "2. Provide supportive care (fluids, nutrition).",
            "3. Contact a veterinarian for antiviral and symptomatic treatment."
        ]
    else:
        advisory = [
            "1. Continue regular health monitoring.",
            "2. Maintain hygiene and vaccination schedules."
        ]
    y = height - 700
    for line in advisory:
        c.drawString(70, y, line)
        y -= 20

    # Disclaimer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(50, 50,
        "Disclaimer: This report is AI-generated for preliminary screening only. Consult a veterinarian for confirmation."
    )

    c.save()
    return filename
