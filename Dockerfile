FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Retrain inside the image so the pickled model matches the installed scikit-learn
RUN python train.py
ENV PORT=8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
