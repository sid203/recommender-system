FROM python:3.8.15-slim-bullseye
WORKDIR /recommender-system
COPY requirements.txt /recommender-system
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
CMD uvicorn app.main:app --reload --workers 1 --host 0.0.0.0 --port $PORT