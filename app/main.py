from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import pytesseract
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import re
import os
import uvicorn


app = FastAPI()

# Set up Tesseract path if needed (adjust if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Make REF_DIR robust regardless of working directory:
REF_DIR = os.path.join(os.path.dirname(__file__), "models", "reference_flags")

# Preprocess and load MobileNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Identity()  # remove classification layer
model.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def normalize_label(label):
    return re.sub(r"\d+$", "", label.lower().replace("_", "")).strip()

from collections import defaultdict

def load_reference_vectors():
    vecs = defaultdict(list)
    for fname in os.listdir(REF_DIR):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        label = normalize_label(os.path.splitext(fname)[0])
        path = os.path.join(REF_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model(tensor).cpu().numpy().flatten()
        vecs[label].append(vec)

    # Average all vectors for same label
    averaged = [(label, np.mean(vlist, axis=0)) for label, vlist in vecs.items()]
    return averaged

ref_vectors = load_reference_vectors()

@app.get("/health")
def health():
    return {"status": "ok", "refs_loaded": len(ref_vectors)}

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def ocr_bottom_region(image, ratio=0.5):
    h = image.shape[0]
    crop = image[int(h * (1 - ratio)):, :]
    text = pytesseract.image_to_string(crop)
    cleaned = re.sub(r"[^A-Za-z ]", "", text).strip().lower()
    return cleaned if len(cleaned) >= 3 else None

def match_label_by_ocr(text):
    for label, _ in ref_vectors:
        if text in label:
            return label
    return None

def get_embedding(img):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(tensor).cpu().numpy().flatten()
    return vec

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # # OCR Step: Only check bottom 50%
    # ocr_result = ocr_bottom_region(img, ratio=0.5)
    # if ocr_result:
    #     ocr_match = match_label_by_ocr(ocr_result)
    #     if ocr_match:
    #         return {
    #             "filename": file.filename,
    #             "prediction": ocr_match,
    #             "confidence": 1.0,
    #             "ocr_detected": ocr_result,
    #             "top3": []
    #         }

    # Fallback to cosine similarity
    query_vec = get_embedding(img)
    scored = [(label, cosine_sim(query_vec, vec)) for label, vec in ref_vectors]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[0]
    top3 = scored[:3]

    return {
        "filename": file.filename,
        "prediction": best[0],
        "confidence": round(best[1], 3),
        # "ocr_detected": ocr_result,
        "top3": [
            {"label": lbl, "score": round(score, 3)} for lbl, score in top3
        ]
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)