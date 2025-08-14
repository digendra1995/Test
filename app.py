from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid
import io

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "mobilenetv2.tflite"
MARKER_DIR = "markers"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Normalize vector
def normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load and embed marker images
marker_embeddings = {}
marker_images = []

def load_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return normalize(output_data[0])

for filename in os.listdir(MARKER_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(MARKER_DIR, filename)
        embedding = load_image_embedding(path)
        marker_embeddings[filename] = embedding
        marker_images.append(filename)

print(f"Loaded {len(marker_embeddings)} markers.")

# Serve marker images
app.mount("/markers", StaticFiles(directory=MARKER_DIR), name="markers")

@app.post("/")
async def match_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        input_embedding = interpreter.get_tensor(output_details[0]['index'])[0]
        input_embedding = normalize(input_embedding)

        best_score = -1
        best_match = None
        log_scores = {}

        for name, embedding in marker_embeddings.items():
            score = cosine_similarity(embedding, input_embedding)
            log_scores[name] = round(float(score), 4)
            print(f"Compared with {name}, similarity: {score:.4f}")
            if score > best_score:
                best_score = score
                best_match = name

        if best_score > 0.85:
            return {
                "marker_id": best_match,
                "animation_url": "",  # optional
                "similarity": round(float(best_score), 4),
                "log": log_scores
            }
        else:
            return {
                "marker_id": None,
                "animation_url": None,
                "similarity": round(float(best_score), 4),
                "log": log_scores
            }

    except Exception as e:
        print("Error processing image:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/markers")
def get_marker_images():
    return [f"/markers/{img}" for img in marker_images]
