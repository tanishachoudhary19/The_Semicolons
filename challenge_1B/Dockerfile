# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files
COPY . .

# Ensure input and output folders exist
RUN mkdir -p /app/input /app/output

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Default command: run main.py (input/output are expected in /app/input and /app/output)
CMD ["python", "main.py"]
