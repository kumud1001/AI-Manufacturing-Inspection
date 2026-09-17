from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL = PROJECT_ROOT / "results" / "quick_test" / "weights" / "best.pt"

SOURCE = (
    PROJECT_ROOT
    / "DataPCB_Final_Clean_6cls"
    / "test"
    / "images"
)

OUTPUT = PROJECT_ROOT / "results" / "predictions"


def predict():
    print("=" * 60)
    print("AI Manufacturing Quality Inspection - Prediction")
    print("=" * 60)

    print(f"Model : {MODEL}")
    print(f"Input : {SOURCE}")

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL}"
        )

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Test images folder not found: {SOURCE}"
        )

    model = YOLO(str(MODEL))

    model.predict(
        source=str(SOURCE),
        imgsz=320,
        conf=0.25,
        device="cpu",

        # Save annotated images
        save=True,

        project=str(OUTPUT),
        name="inspection",
        exist_ok=True
    )

    print("\nPrediction completed.")
    print(f"Results saved to:")
    print(OUTPUT / "inspection")


if __name__ == "__main__":
    predict()