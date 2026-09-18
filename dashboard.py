import streamlit as st
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "results" / "quick_test" / "weights" / "best.pt"
HISTORY = ROOT / "inspection_history.csv"

model = YOLO(MODEL)

st.title("AI Manufacturing Quality Inspection")
st.write("Upload a PCB image for automated defect detection.")


# Dashboard Overview
st.subheader("Inspection Overview")

if HISTORY.exists():
    overview = pd.read_csv(HISTORY)

    total_inspections = len(overview)
    defect_inspections = (overview["Status"] == "DEFECT DETECTED").sum()
    passed_inspections = (overview["Status"] == "PASS").sum()

    if total_inspections > 0:
        defect_rate = (defect_inspections / total_inspections) * 100
    else:
        defect_rate = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Inspections", total_inspections)

    with col2:
        st.metric("Defect Inspections", defect_inspections)

    with col3:
        st.metric("Passed Inspections", passed_inspections)

    with col4:
        st.metric("Defect Rate", f"{defect_rate:.1f}%")
else:
    st.info("No inspection data available yet.")


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

    if defect_count == 0:
        status = "PASS"
    else:
        status = "DEFECT DETECTED"

    st.subheader("Inspection Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Defect Count", defect_count)

    with col2:
        if status == "PASS":
            st.success(status)
        else:
            st.error(status)

    if defect_count > 0:
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

        st.subheader("Detected Defects")
        st.table(display_df)

        st.subheader("Defect Distribution")
        st.bar_chart(df["Defect Type"].value_counts())

        defect_types = ", ".join(df["Defect Type"].unique())

    else:
        defect_types = "None"

    # Save inspection history
    new_record = pd.DataFrame([{
        "Image": file.name,
        "Status": status,
        "Defect Count": defect_count,
        "Defect Types": defect_types
    }])

    if HISTORY.exists():
        old_history = pd.read_csv(HISTORY)
        history = pd.concat(
            [old_history, new_record],
            ignore_index=True
        )
    else:
        history = new_record

    history.to_csv(HISTORY, index=False)

st.subheader("Inspection History")

if HISTORY.exists():
    history = pd.read_csv(HISTORY)
    st.dataframe(history, use_container_width=True)
else:
    st.info("No inspection history yet.")