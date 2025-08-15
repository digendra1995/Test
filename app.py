import os
import torch
import open_clip
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device)
model.eval()

# Load and preprocess marker images
MARKER_FOLDER = "markers"
MARKERS = []

def load_markers():
    global MARKERS
    for filename in os.listdir(MARKER_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(MARKER_FOLDER, filename)
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
            MARKERS.append({"filename": filename, "embedding": embedding.cpu().numpy()})

load_markers()

@app.post("/")
async def match_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            input_embedding = model.encode_image(input_tensor).cpu().numpy()

        best_match = None
        best_distance = float("inf")

        for marker in MARKERS:
            distance = 1 - cosine_similarity(input_embedding, marker["embedding"])[0][0]  # cosine distance
            print(f"Compared with {marker['filename']}, distance: {distance:.4f}")
            if distance < best_distance:
                best_distance = distance
                best_match = marker

        # Set a matching threshold
        if best_distance < 0.3:  # Adjust as needed
            return {"matched": True, "marker": best_match["filename"], "distance": best_distance}
        else:
            return {"matched": False, "distance": best_distance}

    except Exception as e:
        print("Error processing image:", e)
        return {"error": str(e)}
