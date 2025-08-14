import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from PIL import Image
import tensorflow as tf
import faiss
import uuid
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "mobilenetv2.tflite"
MARKER_DIR = "markers"
TEMP_DIR = "temp"
ANIMATIONS = {
    "Shree_Ganesh_Marker_01.jpg": "https://assets6.lottiefiles.com/packages/lf20_jsgzvgrn.json",
    "Shree_Ganesh_Marker_02.jpg": "https://assets10.lottiefiles.com/packages/lf20_ydo1amjm.json"
}

# Ensure temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and process marker images
def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    array = np.array(image).astype(np.float32)
    array = array / 255.0
    return np.expand_dims(array, axis=0)

def extract_feature(img: Image.Image):
    input_tensor = preprocess(img)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

marker_vectors = []
marker_names = []

for filename in os.listdir(MARKER_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(MARKER_DIR, filename)
        with Image.open(image_path).convert('RGB') as img:
            vec = extract_feature(img)
            marker_vectors.append(vec)
            marker_names.append(filename)

if marker_vectors:
    marker_vectors = np.array(marker_vectors).astype(np.float32)
    index = faiss.IndexFlatL2(marker_vectors.shape[1])
    index.add(marker_vectors)

@app.post("/")
async def match_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        filename = f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        temp_path = os.path.join(TEMP_DIR, filename)

        with open(temp_path, "wb") as f:
            f.write(contents)

        image = Image.open(temp_path).convert("RGB")
        vec = extract_feature(image).astype(np.float32)

        if len(marker_vectors) == 0:
            return JSONResponse(content={"error": "No marker vectors loaded."}, status_code=500)

        D, I = index.search(np.array([vec]), k=1)
        best_match_idx = I[0][0]
        distance = D[0][0]

        print(f"Compared with {marker_names[best_match_idx]}, distance: {distance:.4f}")
        print(f"Input vector: {vec[:5]}")
        print(f"Marker vector: {marker_vectors[best_match_idx][:5]}")

        if distance < 30.0:
            return {
                "marker_id": marker_names[best_match_idx],
                "animation_url": ANIMATIONS.get(marker_names[best_match_idx], "")
            }
        else:
            return {"match": False}
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(content={"error": "Failed to process image."}, status_code=500)

@app.get("/markers")
def list_markers():
    try:
        files = [f for f in os.listdir(MARKER_DIR) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        urls = [f"/markers/{name}" for name in files]
        return {"markers": urls}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
