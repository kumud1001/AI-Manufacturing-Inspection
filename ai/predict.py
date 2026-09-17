from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

model = YOLO(ROOT / "results" / "quick_test" / "weights" / "best.pt")

images = list(
    (ROOT / "DataPCB_Final_Clean_6cls" / "test" / "images").glob("*.jpg")
)[:10]

model.predict(
    source=images,
    imgsz=320,
    conf=0.25,
    device="cpu",
    save=True,
    project=ROOT / "results" / "predictions",
    name="test10",
    exist_ok=True
)

print("10-image prediction completed.")