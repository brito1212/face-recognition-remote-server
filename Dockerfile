# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Install system dependencies for face_recognition and OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY utils/ utils/
COPY server.py .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 5000

# Set environment variable for production
ENV FLASK_ENV=production

# Run the server
CMD ["python", "server.py"]
