# Base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy requirements first (for faster build)
COPY requirements-prod.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy only required folders/files
COPY app/ app/
COPY src/ src/
COPY configs/ configs/

# Expose port
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]