import torch
import onnxruntime
import numpy as np
import os
import re
from flask import Flask, render_template, request, redirect
from torchvision import transforms
from PIL import Image

# Encoding dictionary
encoding = {0: 'No_Dr', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}

app = Flask(__name__)

# Create an "uploads" folder if it doesn't exist
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load ONNX Model
onnx_model_path = "efficientnet_b4_best.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize image
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def preprocess_image(image_path):
    """ Load and preprocess image """
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.numpy()

def predict(image_path):
    """ Run image through ONNX model and return predicted label """
    input_tensor = preprocess_image(image_path)
    inputs = {session.get_inputs()[0].name: input_tensor}
    outputs = session.run(None, inputs)
    predicted_class = np.argmax(outputs[0])
    return encoding[predicted_class]

def extract_original_label(filename):

    temp=filename.split('.')[0][-1]
    return encoding[int(temp)]
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)  # Save image

            # Extract original label from filename
            original_label = extract_original_label(file.filename)

            # Get model prediction
            predicted_label = predict(file_path)
            
            return render_template("index.html", 
                                   image_url=file_path, 
                                   image_name=file.filename,  
                                   original_label=original_label,  # Pass extracted label
                                   predicted_label=predicted_label)

    return render_template("index.html", image_url=None, image_name=None, original_label=None, predicted_label=None)

if __name__ == "__main__":
    app.run(debug=True)
