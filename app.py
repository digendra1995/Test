import os
import io
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the lightweight RN50 CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("RN50", pretrained="openai", device=device)
tokenizer = open_clip.get_tokenizer("RN50")

# Directory to store markers
MARKERS_DIR = "markers"
os.makedirs(MARKERS_DIR, exist_ok=True)

# Load all marker vectors
marker_vectors = []
marker_names = []

for filename in os.listdir(MARKERS_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(MARKERS_DIR, filename)
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            marker_vector = model.encode_image(image)
        marker_vectors.append(marker_vector.cpu().numpy())
        marker_names.append(filename)

if marker_vectors:
    marker_vectors = np.vstack(marker_vectors)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/")
async def match_marker(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = image.resize((224, 224))  # Reduce size to save memory
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            input_vector = model.encode_image(image_tensor).cpu().numpy()

        # Compare with loaded markers
        if marker_vectors is not None:
            similarities = cosine_similarity(input_vector, marker_vectors)[0]
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            print(f"Compared with {marker_names[best_idx]}, similarity: {best_score:.4f}")
            print(f"Input vector (first 5): {input_vector[0][:5]}")
            print(f"Marker vector (first 5): {marker_vectors[best_idx][:5]}")

            match = best_score > 0.30  # Adjust threshold as needed
            return JSONResponse({
                "match": match,
                "marker": marker_names[best_idx] if match else None,
                "similarity": round(best_score, 4)
            })

        return JSONResponse({"error": "No marker vectors loaded."}, status_code=500)

    except Exception as e:
        print("Error processing image:", e)
        return JSONResponse({"error": str(e)}, status_code=500)
