from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mobilenetv2.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to extract feature vector from image
def extract_features(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32)
    image = image / 127.5 - 1.0  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    norm = np.linalg.norm(output_data)
    if norm == 0:
        return output_data
    return output_data / norm

# Load all markers from markers folder
MARKERS_DIR = "markers"
marker_features = {}
marker_images = {}

for file in os.listdir(MARKERS_DIR):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(MARKERS_DIR, file)
        with Image.open(path).convert("RGB") as img:
            vec = extract_features(img)
            marker_features[file] = vec
            marker_images[file] = path
        print(f"Loaded marker: {file}")

@app.get("/markers")
def get_marker_images():
    return JSONResponse([f"/marker_image/{name}" for name in marker_images.keys()])

@app.get("/marker_image/{filename}")
def get_marker_image(filename: str):
    path = marker_images.get(filename)
    if path and os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "Image not found"}, status_code=404)

@app.post("/")
async def match_marker(file: UploadFile = File(...)):
    try:
        # Save uploaded frame
        frame_id = str(uuid.uuid4())[:8]
        saved_path = f"received_frames/frame_{frame_id}.jpg"
        os.makedirs("received_frames", exist_ok=True)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with Image.open(saved_path).convert("RGB") as img:
            input_vector = extract_features(img)

        best_match = None
        best_similarity = -1
        threshold = 0.85

        for name, marker_vector in marker_features.items():
            similarity = np.dot(input_vector, marker_vector)
            print(f"Comparing with {name}, similarity: {similarity:.4f}")
            print(f"Input Vector: {input_vector[:5]}... Marker Vector: {marker_vector[:5]}...")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        if best_similarity >= threshold:
            return JSONResponse({"match": best_match, "similarity": float(best_similarity)})
        else:
            return JSONResponse({"match": None, "similarity": float(best_similarity)})

    except Exception as e:
        print("Error processing image:", e)
        return JSONResponse({"error": str(e)}, status_code=500)
