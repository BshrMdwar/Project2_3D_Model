import json
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from manual_test import process_model

app = FastAPI(title="Prediction Server")

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

class PredictRequest(BaseModel):
    model_id: str


@app.post("/predict")
def run_predict(req: PredictRequest):
    result = process_model(req.model_id)
    return result

@app.get("/health")
def health():
    return {"status": "ok"}