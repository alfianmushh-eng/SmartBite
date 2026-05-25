from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException
from smartbite.api.server import FoodQualityAPI
import cv2
import numpy as np


def create_fastapi_app(api: FoodQualityAPI) -> FastAPI:
    app = FastAPI(title="SmartBite Food Quality API", version="1.0.0")

    @app.get("/health")
    async def health():
        return api.health()

    @app.post("/analyze")
    async def analyze(file: UploadFile = File(...)):
        if not file.content_type or "image" not in file.content_type:
            raise HTTPException(400, "Image file required")
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Invalid image data")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return api.analyze(img)

    @app.post("/analyze/batch")
    async def analyze_batch(files: list[UploadFile] = File(...)):
        results = []
        for f in files:
            contents = await f.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results.append(api.analyze(img))
        return {"results": results, "count": len(results)}

    return app
