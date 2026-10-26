"""FastAPI service that assigns customers to marketing segments."""
import io
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
STATIC_DIR = Path(__file__).resolve().parent / "static"

pipeline = joblib.load(MODEL_DIR / "kmeans_pipeline.joblib")
meta = json.loads((MODEL_DIR / "segments.json").read_text())
FEATURES: list[str] = meta["features"]
SEGMENTS = {s["id"]: s for s in meta["segments"]}

app = FastAPI(
    title="Customer Segmentation API",
    description="K-Means segmentation of retail customers for marketing analysis.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Customer(BaseModel):
    age: int = Field(..., ge=18, le=100, examples=[35])
    income: float = Field(..., ge=0, examples=[85000])
    spending_score: float = Field(..., ge=1, le=100, examples=[70])
    membership_years: int = Field(..., ge=0, le=50, examples=[6])
    purchase_frequency: int = Field(..., ge=0, examples=[30])
    last_purchase_amount: float = Field(..., ge=0, examples=[650.0])


class Prediction(BaseModel):
    segment_id: int
    segment_name: str
    description: str
    top_category: str
    distance_to_center: float


def predict_frame(df: pd.DataFrame) -> list[Prediction]:
    X = df[FEATURES]
    labels = pipeline.predict(X)
    distances = pipeline.transform(X).min(axis=1)
    return [
        Prediction(
            segment_id=int(label),
            segment_name=SEGMENTS[int(label)]["name"],
            description=SEGMENTS[int(label)]["description"],
            top_category=SEGMENTS[int(label)]["top_category"],
            distance_to_center=round(float(dist), 3),
        )
        for label, dist in zip(labels, distances)
    ]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/segments")
def segments():
    """Profiles of every segment learned from the training data."""
    return {
        "n_customers": meta["n_customers"],
        "silhouette": meta["silhouette"],
        "overall_mean": meta["overall_mean"],
        "segments": meta["segments"],
    }


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    """Assign a single customer to a segment."""
    return predict_frame(pd.DataFrame([customer.model_dump()]))[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(customers: list[Customer]):
    """Assign many customers (JSON list) to segments."""
    if not customers:
        raise HTTPException(400, "Provide at least one customer.")
    return predict_frame(pd.DataFrame([c.model_dump() for c in customers]))


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Upload a CSV with the feature columns; returns each row with its segment."""
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not parse the file as CSV.")
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {', '.join(missing)}")
    df = df.dropna(subset=FEATURES)
    if df.empty:
        raise HTTPException(400, "No complete rows to score.")
    preds = predict_frame(df)
    df["segment_id"] = [p.segment_id for p in preds]
    df["segment_name"] = [p.segment_name for p in preds]
    counts = df["segment_name"].value_counts().to_dict()
    return {"rows": len(df), "counts": counts, "preview": df.head(50).to_dict(orient="records")}
