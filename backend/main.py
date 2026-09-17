from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil

app = FastAPI(title="AI Manufacturing Quality Inspection")

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "results" / "quick_test" / "weights" / "best.pt"
UPLOADS = ROOT / "uploads"

UPLOADS.mkdir(exist_ok=True)

model = YOLO(MODEL)


@app.get("/")
def home():
    return {"message": "AI Manufacturing Inspection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = UPLOADS / file.filename

    with open(image, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(
        source=image,
        imgsz=320,
        conf=0.25,
        device="cpu"
    )

    detections = []

    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "box": [round(float(x), 2) for x in box.xyxy[0]]
            })

    return {
        "filename": file.filename,
        "detections": detections
    }