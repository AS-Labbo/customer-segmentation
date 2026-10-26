# Customer Segmentation API

K-Means customer segmentation for marketing analysis, served with **FastAPI**.
It includes a web page for exploring the segments, classifying a single customer and scoring a CSV file.

Dataset: [Customer Segmentation Data for Marketing Analysis](https://www.kaggle.com/datasets/fahmidachowdhury/customer-segmentation-data-for-marketing-analysis) (1,000 customers).

## Segments

| Segment | Customers | What sets them apart |
|---|---|---|
| Loyal High Spenders | 234 | Long membership, high spending score, large purchases |
| New Big-Ticket Buyers | 255 | Recent members who make large purchases |
| Cautious Low Spenders | 266 | Low spending score and small purchases |
| Engaged Small-Basket Shoppers | 245 | High spending score but small purchases, newer members |

Features used: `age`, `income`, `spending_score`, `membership_years`, `purchase_frequency`, `last_purchase_amount` (standardised, then K-Means with k=4).

> **A note on the data:** the dataset looks synthetic, with features spread almost uniformly, so the clusters are not sharply separated (silhouette ≈ 0.13 for every k from 2 to 8). The segments are a useful way to split customers for marketing, not natural groups.

## Project structure

```
app/main.py              FastAPI app (API + web page)
app/static/index.html    Web page
train.py                 Trains the model, writes model/
model/                   Trained pipeline + segment profiles
data/                    Dataset CSV
tests/test_api.py        API tests
```

## Run locally

```bash
pip install -r requirements-dev.txt
python train.py                 # add --download to fetch the data from Kaggle again
uvicorn app.main:app --reload
```

Open http://localhost:8000 for the web page and http://localhost:8000/docs for the interactive API docs.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/segments` | Segment profiles |
| POST | `/predict` | Classify one customer (JSON) |
| POST | `/predict/batch` | Classify a list of customers (JSON) |
| POST | `/predict/csv` | Upload a CSV and get each row's segment |
| GET | `/health` | Health check |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age":35,"income":85000,"spending_score":70,"membership_years":6,"purchase_frequency":30,"last_purchase_amount":650}'
```

## Deploy

GitHub hosts the code; the API runs on a host that can run Python.

**Render (free):** on render.com choose **New → Blueprint**, pick this repository, and it uses `render.yaml`. The model is retrained during the build so it always matches the installed scikit-learn.

**Docker (Railway, Fly.io, Hugging Face Spaces, …):**

```bash
docker build -t customer-segmentation .
docker run -p 8000:8000 customer-segmentation
```

GitHub Actions runs the tests on every push (`.github/workflows/ci.yml`).
