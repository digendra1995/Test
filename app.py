from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "mobilenetv2.tflite"
MARKER_FOLDER = "markers"
TEMP_INPUTS = "inputs"
os.makedirs(TEMP_INPUTS, exist_ok=True)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

IMG_SIZE = (224, 224)

def extract_vector(image: Image.Image):
    image = image.resize(IMG_SIZE).convert('RGB')
    img_array = np.array(image).astype(np.float32) / 255.0
    input_data = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Normalize the vector
    norm = np.linalg.norm(output_data)
    if norm != 0:
        output_data = output_data / norm

    return output_data

# Load marker vectors
marker_vectors = []
marker_names = []
print("Loading markers:")
for file_name in os.listdir(MARKER_FOLDER):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(MARKER_FOLDER, file_name)
        img = Image.open(path)
        vec = extract_vector(img)
        marker_vectors.append(vec)
        marker_names.append(file_name)
        print(f"Loaded {file_name}")

marker_vectors = np.array(marker_vectors)

@app.post("/")
async def match_marker(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(tf.io.gfile.GFile(file.filename, 'rb') if hasattr(file, 'filename') else file.file)

        # Save input image for review
        input_id = str(uuid.uuid4()) + ".jpg"
        input_path = os.path.join(TEMP_INPUTS, input_id)
        with open(input_path, 'wb') as f:
            f.write(contents)

        vec = extract_vector(img)

        # Compute cosine similarity
        similarities = cosine_similarity([vec], marker_vectors)[0]
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        # Log for debugging
        print(f"Compared with {marker_names[best_idx]}, similarity: {best_score:.4f}")
        print(f"Input vector (first 5): {vec[:5]}")
        print(f"Marker vector (first 5): {marker_vectors[best_idx][:5]}")

        if best_score > 0.85:
            return {"matched": True, "marker_id": marker_names[best_idx]}
        else:
            return {"matched": False}

    except Exception as e:
        print("Error processing image:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
