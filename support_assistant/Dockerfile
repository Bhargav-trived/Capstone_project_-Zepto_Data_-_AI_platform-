FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    MOCK_LLM=1 \
    PORT=7860

WORKDIR /app

# Install build dependencies for ChromaDB native bindings
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SentenceTransformer all-MiniLM-L6-v2 during build
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app.py .

# Pre-ingest ChromaDB vectors during image build
RUN python3 -c "import app; app.ensure_docs_exist(); app.get_vectorstore()"

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]