from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

model = YOLO(
    ROOT / "results" / "quick_test" / "weights" / "best.pt"
)

model.val(
    data=ROOT / "ai" / "dataset.yaml",
    imgsz=320,
    batch=8,
    device="cpu",
    plots=True,
    project=ROOT / "results",
    name="evaluation",
    exist_ok=True
)

print("Evaluation completed.")