from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import torch
import open_clip as clip
import os
import io

app = FastAPI()

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load marker images and generate their features
MARKER_DIR = "markers"
marker_features = []
marker_names = []

def load_markers():
    marker_features.clear()
    marker_names.clear()
    for filename in os.listdir(MARKER_DIR):
        path = os.path.join(MARKER_DIR, filename)
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model.encode_image(image)
                feature /= feature.norm(dim=-1, keepdim=True)
            marker_features.append(feature.cpu())
            marker_names.append(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

load_markers()

@app.post("/")
async def match_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            input_feature = model.encode_image(input_tensor)
            input_feature /= input_feature.norm(dim=-1, keepdim=True)

        # Compare with marker features
        similarities = [
            cosine_similarity(input_feature.cpu().numpy(), marker.cpu().numpy())[0][0]
            for marker in marker_features
        ]

        # Log distances
        for name, score in zip(marker_names, similarities):
            print(f"Compared with {name}, similarity: {score:.4f}")

        # Best match
        best_idx = int(torch.argmax(torch.tensor(similarities)))
        best_score = similarities[best_idx]
        threshold = 0.80  # CLIP scores are usually between 0.0 and 1.0

        if best_score >= threshold:
            return JSONResponse({
                "match": True,
                "marker": marker_names[best_idx],
                "similarity": best_score
            })
        else:
            return JSONResponse({
                "match": False,
                "similarity": best_score
            })

    except Exception as e:
        return JSONResponse(content={"error": f"Error processing image: {e}"}, status_code=500)
