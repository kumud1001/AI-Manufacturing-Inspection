import streamlit as st
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "results" / "quick_test" / "weights" / "best.pt"

model = YOLO(MODEL)

st.title("AI Manufacturing Quality Inspection")
st.write("Upload a PCB image for automated defect detection.")

file = st.file_uploader(
    "Upload PCB Image",
    type=["jpg", "jpeg", "png"]
)

if file:
    upload_dir = ROOT / "uploads"
    upload_dir.mkdir(exist_ok=True)

    image_path = upload_dir / file.name

    with open(image_path, "wb") as f:
        f.write(file.getbuffer())

    results = model.predict(
        source=image_path,
        imgsz=320,
        conf=0.25,
        device="cpu"
    )

    result = results[0]

    st.image(image_path, caption="PCB Image")

    if len(result.boxes) == 0:
        st.success("PASS - No defect detected")
    else:
        st.error(f"DEFECT DETECTED - {len(result.boxes)} defect(s)")

        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            st.write(
                f"**{name}** — Confidence: {confidence:.2%}"
            )