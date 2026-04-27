# 1. Use an official, lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system-level dependencies required by OpenCV and YOLO
# (Even though we use the headless version, these prevent Linux crash errors)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your requirements file into the container
COPY requirements.txt .

# 5. Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy all your code, images, and the YOLO model into the container
COPY . .

# 7. Railway dynamically assigns a port, so we tell Uvicorn to listen to it
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]