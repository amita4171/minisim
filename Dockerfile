FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p results

# Expose both API and dashboard ports
EXPOSE 8000 8501

# Default: run FastAPI production API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
