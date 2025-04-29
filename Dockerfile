# Use Python 3.9 as base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper model
RUN python -c "import whisper; whisper.load_model('base')"

# Copy application code
COPY . .

# Create directory for temporary files
RUN mkdir -p /tmp

# Run as non-root user
RUN useradd -m appuser
RUN chown -R appuser:appuser /app /tmp
USER appuser

# Expose port
EXPOSE 8080

# Run the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "300", "app:app"]