import os
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import open_clip
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Allow all CORS (frontend-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("RN50", pretrained="openai")
tokenizer = open_clip.get_tokenizer("RN50")
model.to(device).eval()

# Load marker vectors
marker_dir = "markers"
marker_vectors = []

def encode_image(img: Image.Image):
    img = img.convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(img)
        vec /= vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()

for filename in os.listdir(marker_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(marker_dir, filename)
        img = Image.open(path)
        vec = encode_image(img)
        marker_vectors.append((filename, vec))
        print(f"Loaded marker: {filename}")

@app.post("/")
async def match_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        temp_path = f"temp_{uuid.uuid4()}.jpg"
        with open(temp_path, "wb") as f:
            f.write(contents)

        img = Image.open(temp_path).convert("RGB")
        input_vec = encode_image(img)

        os.remove(temp_path)

        best_match = None
        best_score = -1

        for marker_name, marker_vec in marker_vectors:
            score = cosine_similarity(input_vec, marker_vec)[0][0]
            print(f"Compared with {marker_name}, similarity: {score:.4f}")
            print("Input vector:", input_vec[0][:5])
            print("Marker vector:", marker_vec[0][:5])

            if score > best_score:
                best_score = score
                best_match = marker_name

        THRESHOLD = 0.90  # Tune this
        if best_score >= THRESHOLD:
            return {
                "match": True,
                "marker": best_match,
                "score": float(best_score)
            }
        else:
            return {
                "match": False,
                "score": float(best_score)
            }

    except Exception as e:
        print("Error processing image:", str(e))
        return {"error": str(e)}
