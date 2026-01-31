# High-performance RUL Prediction API
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create logs directory
RUN mkdir -p logs

# Expose port
EXPOSE 8000

# Set environment variables for performance
ENV PYTHONUNBUFFERED=1
ENV WORKERS=4
ENV MAX_REQUESTS=1000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.true_rul.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]