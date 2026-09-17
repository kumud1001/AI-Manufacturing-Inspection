from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid

from ai.predict import QualityInspector


app = FastAPI(
    title="AI Manufacturing Quality Inspection API",
    version="1.0.0"
)


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


inspector = QualityInspector()


@app.get("/")
def home():

    return {
        "system": "AI Manufacturing Quality Inspection",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/inspect")
async def inspect_product(
    file: UploadFile = File(...)
):

    file_id = str(uuid.uuid4())

    file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = inspector.inspect(
        str(file_path)
    )

    return JSONResponse({
        "file": file.filename,
        "inspection": result
    })