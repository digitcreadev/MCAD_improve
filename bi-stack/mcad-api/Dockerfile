FROM python:3.11-slim

WORKDIR /app

# 1) Install real backend dependencies (requirements.txt is at repo root backend/)
COPY backend/requirements.txt /tmp/backend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt

# 2) Ensure the API runtime deps are present (in case requirements doesn't include them)
RUN pip install --no-cache-dir fastapi uvicorn pydantic requests pyyaml neo4j networkx

# 3) Copy the bi-stack mcad-api app (because build context is repo root)
COPY bi-stack/mcad-api/app.py /app/app.py

# 4) Folder for logs/artefacts (fallback if no Neo4j)
RUN mkdir -p /app/data

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend:/app

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]