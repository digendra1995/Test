import os
import io
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import open_clip
from torchvision import transforms

# App init
app = FastAPI()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folders if not exist
os.makedirs("inputs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("markers", exist_ok=True)

# Load CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Load and encode all marker images
marker_vectors: Dict[str, torch.Tensor] = {}
def load_marker_vectors():
    print("Loading marker vectors...")
    for filename in os.listdir("markers"):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join("markers", filename)
            image = Image.open(path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                vec = model.encode_image(image_input)
                vec /= vec.norm(dim=-1, keepdim=True)
                marker_vectors[filename] = vec.cpu().numpy()[0]
    print(f"Loaded {len(marker_vectors)} markers.")

load_marker_vectors()

# Helper to get image embedding
def get_clip_vector(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image_input)
        vec /= vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0]

@app.post("/")
async def match_marker(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Save input image for inspection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = f"inputs/input_{timestamp}.jpg"
        img.save(input_path)

        # Get vector
        input_vector = get_clip_vector(img)

        # Compare with markers
        best_match = None
        best_score = -1

        for name, marker_vec in marker_vectors.items():
            score = cosine_similarity([input_vector], [marker_vec])[0][0]
            print(f"Compared with {name}, similarity: {score:.4f}")

            if score > best_score:
                best_score = score
                best_match = name

        threshold = 0.85
        matched = best_score >= threshold

        if matched:
            print(f"✅ Match found: {best_match} ({best_score:.4f})")
        else:
            print(f"❌ No match found. Closest: {best_match} ({best_score:.4f})")

        return {
            "matched": matched,
            "marker": best_match if matched else None,
            "score": round(float(best_score), 4),
            "input_vector_preview": input_vector[:5].tolist(),
            "marker_vector_preview": marker_vectors[best_match][:5].tolist() if best_match else []
        }

    except Exception as e:
        print("Error processing image:", e)
        return {"error": str(e)}
