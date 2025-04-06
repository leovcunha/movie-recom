FROM node:20-slim AS frontend-builder

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# Build frontend in production mode
ENV NODE_ENV=production
RUN npm run build
RUN echo "Contents of build:" && ls -la src/main/resources/static/

FROM python:3.10.13-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gdown

# Create static directory and copy the built frontend
WORKDIR /app
RUN mkdir -p /app/static/built
# Copy the frontend build
COPY --from=frontend-builder /app/src/main/resources/static/built /app/static/built
COPY --from=frontend-builder /app/src/main/resources/static/index.html /app/static/index.html
RUN echo "Contents of static:" && ls -la static/

# Copy backend code
COPY src/backend ./
RUN mkdir -p data && \
    gdown --id 1KNHvgPM8HupZl6XNrd5in0Cw6WL3-5FW -O data/ratings_tmdb.parquet && \
    ls -la data/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV ENV=production

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"] 