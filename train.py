"""Train a K-Means customer segmentation model and save it to model/.

Usage:
    python train.py            # uses data/customer_segmentation_data.csv
    python train.py --download # fetches the dataset from Kaggle first
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "customer_segmentation_data.csv"
MODEL_DIR = ROOT / "model"

FEATURES = [
    "age",
    "income",
    "spending_score",
    "membership_years",
    "purchase_frequency",
    "last_purchase_amount",
]
N_CLUSTERS = 4


def download_dataset() -> None:
    import shutil

    import kagglehub

    path = Path(kagglehub.dataset_download(
        "fahmidachowdhury/customer-segmentation-data-for-marketing-analysis"
    ))
    DATA_PATH.parent.mkdir(exist_ok=True)
    shutil.copy(path / "customer_segmentation_data.csv", DATA_PATH)


def describe_segment(center: pd.Series, overall: pd.Series, spread: pd.Series) -> tuple[str, str]:
    """Name a segment from how far its centre sits from the overall mean (in std units)."""
    z = (center - overall) / spread
    score, basket, tenure = z["spending_score"], z["last_purchase_amount"], z["membership_years"]

    if tenure > 0.5 and score > 0.3:
        name = "Loyal High Spenders"
    elif basket > 0.5 and tenure < 0:
        name = "New Big-Ticket Buyers"
    elif score < -0.5 and basket < -0.5:
        name = "Cautious Low Spenders"
    elif score > 0.5 and basket < -0.3:
        name = "Engaged Small-Basket Shoppers"
    elif z["income"] > 0.5:
        name = "Affluent Occasional Buyers"
    else:
        name = "Average Customers"

    top = z.abs().sort_values(ascending=False).index[:3]
    traits = ", ".join(f"{'higher' if z[f] > 0 else 'lower'} {f.replace('_', ' ')}" for f in top)
    return name, f"Compared with the average customer: {traits}."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--clusters", type=int, default=N_CLUSTERS)
    args = parser.parse_args()

    if args.download or not DATA_PATH.exists():
        download_dataset()

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=args.clusters, n_init=20, random_state=42)),
    ])
    labels = pipe.fit_predict(X)
    df["segment"] = labels

    scores = {}
    scaled = pipe.named_steps["scaler"].transform(X)
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
        scores[k] = round(float(silhouette_score(scaled, km.labels_)), 4)

    overall, spread = X.mean(), X.std()
    centers = df.groupby("segment")[FEATURES].mean()
    used_names: set[str] = set()
    segments = []
    for seg_id, center in centers.iterrows():
        name, description = describe_segment(center, overall, spread)
        if name in used_names:
            name = f"{name} ({seg_id})"
        used_names.add(name)
        members = df[df.segment == seg_id]
        segments.append({
            "id": int(seg_id),
            "name": name,
            "description": description,
            "size": int(len(members)),
            "share": round(len(members) / len(df), 4),
            "profile": {f: round(float(center[f]), 2) for f in FEATURES},
            "top_category": members.preferred_category.mode().iat[0],
            "gender_mix": members.gender.value_counts(normalize=True).round(3).to_dict(),
        })

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "kmeans_pipeline.joblib")
    meta = {
        "features": FEATURES,
        "n_clusters": args.clusters,
        "n_customers": int(len(df)),
        "silhouette": scores[args.clusters],
        "silhouette_by_k": scores,
        "overall_mean": {f: round(float(overall[f]), 2) for f in FEATURES},
        "segments": segments,
    }
    (MODEL_DIR / "segments.json").write_text(json.dumps(meta, indent=2))

    print(f"Trained {args.clusters} segments on {len(df)} customers, silhouette={scores[args.clusters]}")
    for s in segments:
        print(f"  [{s['id']}] {s['name']:<30} {s['size']:>4} customers  {s['description']}")


if __name__ == "__main__":
    main()
