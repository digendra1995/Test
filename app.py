from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid

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

# Create marker embeddings
def load_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

marker_embeddings = {}
marker_images = []

for filename in os.listdir(MARKER_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(MARKER_DIR, filename)
        embedding = load_image_embedding(path)
        marker_embeddings[filename] = embedding
        marker_images.append(filename)

print(f"Loaded {len(marker_embeddings)} markers.")

# Serve static marker images
app.mount("/markers", StaticFiles(directory=MARKER_DIR), name="markers")

# Main match endpoint
@app.post("/")
async def match_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(tf.io.BytesIO(contents)).convert("RGB").resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        input_embedding = interpreter.get_tensor(output_details[0]['index'])[0]

        best_match = None
        best_distance = float('inf')

        for name, embedding in marker_embeddings.items():
            distance = np.linalg.norm(embedding - input_embedding)
            print(f"Compared with {name}, distance: {distance:.4f}")
            if distance < best_distance:
                best_distance = distance
                best_match = name

        if best_distance < 0.5:
            return {
                "marker_id": best_match,
                "animation_url": "",  # Optional if using animation
                "distance": round(float(best_distance), 4)
            }
        else:
            return {
                "marker_id": None,
                "animation_url": None,
                "distance": round(float(best_distance), 4)
            }

    except Exception as e:
        print("Error processing image:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# Endpoint to list marker image URLs
@app.get("/markers")
def get_marker_images():
    return [f"/markers/{img}" for img in marker_images]
