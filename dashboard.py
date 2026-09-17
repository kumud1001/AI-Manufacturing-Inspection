import streamlit as st
import pandas as pd
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
    annotated = result.plot()

    st.image(
        annotated,
        caption="Inspection Result",
        channels="BGR"
    )

    defect_count = len(result.boxes)

    st.subheader("Inspection Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Defect Count", defect_count)

    with col2:
        if defect_count == 0:
            st.success("PASS")
        else:
            st.error("DEFECT DETECTED")

    if defect_count > 0:
        st.subheader("Detected Defects")

        data = []

        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            data.append({
                "Defect Type": name,
                "Confidence": confidence
            })

        df = pd.DataFrame(data)

        display_df = df.copy()
        display_df["Confidence"] = display_df["Confidence"].map(
            lambda x: f"{x:.2%}"
        )

        st.table(display_df)

        st.subheader("Defect Distribution")

        counts = df["Defect Type"].value_counts()

        st.bar_chart(counts)