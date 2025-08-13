from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import faiss
import io
import os

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite MobileNetV2 model
MODEL_PATH = "mobilenet_v2.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Example markers (replace with your actual files)
marker_files = {
    "marker1": "markers/Shree_Ganesh_Marker_01.jpg",
    "marker2": "markers/Shree_Ganesh_Marker_02.jpg"
}
marker_animations = {
    "marker1": "animations/Ganesha_Test01.json",
    "marker2": "animations/Ganesha_Test01.json"
}

# Preprocess function
def preprocess_image(image_source):
    if isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0  # MobileNetV2 normalization
    return np.expand_dims(arr, axis=0)

# Run inference to get embedding
def get_embedding(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Build FAISS index
d = 1280
index = faiss.IndexFlatL2(d)
marker_labels = []

for label, path in marker_files.items():
    arr = preprocess_image(path)
    vec = get_embedding(arr)
    index.add(vec)
    marker_labels.append(label)

@app.post("/match")
async def match_marker(file: UploadFile = File(...)):
    image_bytes = await file.read()
    arr = preprocess_image(image_bytes)
    vec = get_embedding(arr)
    D, I = index.search(vec, 1)
    best_label = marker_labels[I[0][0]]
    return {
        "marker_id": best_label,
        "animation_url": marker_animations[best_label],
        "distance": float(D[0][0])
    }
