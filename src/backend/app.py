from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
from recommendation_service import MovieRecommender  # Changed to absolute import
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update path to be relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data", "ratings_small.csv")

# Initialize recommender once when starting the server
try:
    recommender = MovieRecommender(data_path)
except Exception as e:
    logger.error(f"Failed to initialize recommender: {e}")
    raise


class UserPreferences(BaseModel):
    ratings: Dict[int, float]


@app.post("/recommendations/")
async def get_recommendations(preferences: UserPreferences):
    try:
        # Convert string keys to integers in the preferences
        int_preferences = {int(k): float(v) for k, v in preferences.ratings.items()}
        recommendations = recommender.get_recommendations(int_preferences)

        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found")

        return recommendations
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/valid-movies")
async def get_valid_movies():
    """Return valid movie IDs with some statistics"""
    try:
        # Convert numpy types to Python native types
        all_movie_ids = [int(id) for id in recommender.movie_ids]
        popular_movies = recommender.get_popular_movies(20)

        return {
            "total_movies": len(all_movie_ids),
            "id_range": f"{min(all_movie_ids)} to {max(all_movie_ids)}",
            "popular_movies": {int(k): float(v) for k, v in popular_movies.items()},
            "sample_ids": [int(id) for id in all_movie_ids[:50]],
        }
    except Exception as e:
        logger.error(f"Error getting valid movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
