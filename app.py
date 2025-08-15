import os
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import onnxruntime as ort
import torchvision.transforms as transforms

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX CLIP model
onnx_model_path = "clip-vit-base-patch32.onnx"  # Make sure this is in your project
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# Load and preprocess marker images
marker_folder = "markers"
marker_vectors = []
marker_names = []

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

def image_to_vector(img: Image.Image):
    img_tensor = preprocess(img).unsqueeze(0).numpy().astype(np.float32)
    ort_inputs = {session.get_inputs()[0].name: img_tensor}
    ort_outs = session.run(None, ort_inputs)
    vec = ort_outs[0][0]
    return vec / np.linalg.norm(vec)

# Preload marker vectors
for fname in os.listdir(marker_folder):
    if fname.lower().endswith((".jpg", ".png")):
        img = Image.open(os.path.join(marker_folder, fname)).convert("RGB")
        vector = image_to_vector(img)
        marker_vectors.append(vector)
        marker_names.append(fname)
print(f"Loaded {len(marker_vectors)} markers.")

@app.post("/")
async def compare_image(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        input_vec = image_to_vector(img)

        # Compare with all markers
        best_match = None
        best_score = -1
        for idx, marker_vec in enumerate(marker_vectors):
            score = np.dot(input_vec, marker_vec)  # Cosine similarity
            if score > best_score:
                best_score = score
                best_match = marker_names[idx]

        matched = best_score > 0.75  # Threshold for match

        return {
            "matched": matched,
            "marker": best_match if matched else None,
            "score": float(best_score)
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
