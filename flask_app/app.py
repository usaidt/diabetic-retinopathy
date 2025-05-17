import torch
import onnxruntime
import numpy as np
import os
import re
import time
import random
from flask import Flask, render_template, request, redirect, jsonify
from torchvision import transforms
from PIL import Image

# Encoding dictionary
encoding = {0: 'No_Dr', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}

app = Flask(__name__)

# Create an "uploads" folder if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load ONNX Model
onnx_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "efficientnet_b4_best.onnx")
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
    """ Run image through ONNX model and return predicted label and thinking process """
    try:
        # Simulate model thinking time
        time.sleep(random.uniform(0.5, 2.0))
        
        # Get input tensor
        input_tensor = preprocess_image(image_path)
        input_shape = input_tensor.shape
        input_mean = np.mean(input_tensor)
        input_std = np.std(input_tensor)
        
        # Run model
        inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, inputs)
        
        # Get probabilities for all classes
        probabilities = outputs[0][0]
        predicted_class = np.argmax(probabilities)
        
        # Create thinking process data
        thinking_process = {
            "input_analysis": {
                "shape": str(input_shape),
                "mean_value": float(input_mean),
                "std_value": float(input_std)
            },
            "model_output": {
                "probabilities": {
                    encoding[i]: float(prob) for i, prob in enumerate(probabilities)
                },
                "selected_class": int(predicted_class),
                "confidence": float(probabilities[predicted_class])
            }
        }
        
        return encoding[predicted_class], thinking_process
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return "Error during prediction", {"error": str(e)}

def extract_original_label(filename):
    """Extract the label number from the filename (e.g., '10043_right_LABEL_2.jpeg' -> '2')"""
    match = re.search(r'LABEL_(\d+)', filename)
    if match:
        return encoding[int(match.group(1))]
    return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle test image selection
        if "test_image" in request.form:
            filename = request.form["test_image"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            
            # Extract original label from filename
            original_label = extract_original_label(filename)
            
            # Get model prediction and thinking process
            predicted_label, thinking_process = predict(file_path)
            print(f"Processing image: {filename}")
            print(f"Original label: {original_label}")
            print(f"Predicted label: {predicted_label}")
            print(f"Thinking process: {thinking_process}")
            
            return render_template("index.html", 
                                image_url=f"/static/uploads/{filename}",
                                image_name=filename,
                                original_label=original_label,
                                predicted_label=predicted_label,
                                thinking_process=thinking_process)
        
        # Handle file upload
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

            # Get model prediction and thinking process
            predicted_label, thinking_process = predict(file_path)
            print(f"Processing uploaded image: {file.filename}")
            print(f"Original label: {original_label}")
            print(f"Predicted label: {predicted_label}")
            print(f"Thinking process: {thinking_process}")
            
            return render_template("index.html", 
                                image_url=f"/static/uploads/{file.filename}",
                                image_name=file.filename,  
                                original_label=original_label,
                                predicted_label=predicted_label,
                                thinking_process=thinking_process)

    return render_template("index.html", 
                         image_url=None, 
                         image_name=None, 
                         original_label=None, 
                         predicted_label=None,
                         thinking_process=None)

if __name__ == "__main__":
    app.run(debug=True)
