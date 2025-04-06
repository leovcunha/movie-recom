#!/usr/bin/env bash
set -e

# Print all executed commands
set -x

# Download the data file
echo "Downloading data files..."
mkdir -p src/backend/data
curl -L "https://drive.google.com/uc?export=download&id=1KNHvgPM8HupZl6XNrd5in0Cw6WL3-5FW" -o src/backend/data/ratings_tmdb.parquet

# Check if data download was successful
if [ ! -f src/backend/data/ratings_tmdb.parquet ]; then
  echo "Failed to download data file"
  exit 1
fi

echo "Data downloaded successfully"
ls -la src/backend/data/

# The Docker build process is defined in the Dockerfile
# This script is mainly to handle data download

# If there are any other build steps needed, they would go here
echo "Build script completed successfully" 