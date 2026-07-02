# catalog.json must exist in the build context before `docker build`.
# Generate/refresh it with:  python fetch_catalog.py
# (kept as a separate offline step rather than fetched during the image
# build, so the image is reproducible and doesn't need network access to
# tcp-us-prod-rnd.shl.com at build time.)

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py schemas.py retrieval.py agent.py ./
COPY catalog.json ./

ENV SHL_CATALOG_PATH=/app/catalog.json
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

