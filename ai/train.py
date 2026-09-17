from pathlib import Path
import shutil
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET = PROJECT_ROOT / "ai" / "dataset.yaml"
MODEL = PROJECT_ROOT / "yolo26n.pt"


def train():
    print("=" * 60)
    print("AI Manufacturing Quality Inspection - QUICK TEST")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Dataset file : {DATASET}")
    print(f"Model        : {MODEL}")

    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATASET}")

    if not MODEL.exists():
        raise FileNotFoundError(f"YOLO model not found: {MODEL}")

    # Load pretrained YOLO model
    model = YOLO(str(MODEL))

    # Quick CPU test
    results = model.train(
        data=str(DATASET),

        # Very small test
        epochs=1,
        imgsz=320,
        batch=8,

        # CPU settings
        device="cpu",
        workers=0,

        # Do not wait for many epochs
        patience=1,

        # Save results
        project=str(PROJECT_ROOT / "results"),
        name="quick_test",
        exist_ok=True,

        # Useful for quick testing
        cache=False,
        verbose=True
    )

    print("\n" + "=" * 60)
    print("QUICK TEST TRAINING COMPLETED")
    print("=" * 60)

    print("Results saved to:")
    print(PROJECT_ROOT / "results" / "quick_test")


if __name__ == "__main__":
    train()