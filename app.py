# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import tensorflow as tf
from PIL import Image
import faiss
import io
import os

# Constants
MODEL_PATH = "mobilenetv2.tflite"
MARKER_DIR = "markers"

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Helper: preprocess image to 224x224 RGB normalized
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224)).convert("RGB")
    arr = np.array(image).astype(np.float32)
    arr = arr / 127.5 - 1.0  # Normalize
    return np.expand_dims(arr, axis=0)

# Helper: extract embedding from image
def get_embedding(image: Image.Image) -> np.ndarray:
    input_tensor = preprocess(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output.flatten()

# Load and vectorize all markers
marker_vectors = []
marker_names = []
for filename in os.listdir(MARKER_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(MARKER_DIR, filename)
        img = Image.open(path)
        vec = get_embedding(img)
        marker_vectors.append(vec)
        marker_names.append(filename)

# Build FAISS index
d = len(marker_vectors[0]) if marker_vectors else 1280
index = faiss.IndexFlatL2(d)
if marker_vectors:
    index.add(np.array(marker_vectors))

@app.post("/")
async def match_marker(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    vec = get_embedding(img)

    if not index.ntotal:
        return {"error": "No markers indexed."}

    D, I = index.search(np.array([vec]), k=1)
    best_match_idx = I[0][0]
    distance = D[0][0]

    # Set threshold for match (adjust as needed)
    if distance < 2.0:
        marker_id = marker_names[best_match_idx]
        animation_url = "https://assets2.lottiefiles.com/packages/lf20_puciaact.json"
        return {"marker_id": marker_id, "animation_url": animation_url}
    else:
        return {"marker_id": None, "message": "No match found"}

@app.get("/")
def health_check():
    return {"status": "Backend is running"}
