# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY requirements.txt .
COPY main.py .
COPY rag.py .
COPY .env .

# Copy frontend file
COPY index.html .

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the backend port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
