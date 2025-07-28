FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir PyMuPDF

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Copy the processing script
COPY process_pdfs.py .

CMD ["python", "process_pdfs.py"] 