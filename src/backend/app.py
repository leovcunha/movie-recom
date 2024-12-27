from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict
from recommendation_service import MovieRecommender  # Changed to absolute import
import os
import logging
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug: Print current directory and contents
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Directory contents: {os.listdir('.')}")
logger.debug(f"Static directory contents: {os.listdir('static') if os.path.exists('static') else 'static directory not found'}")

# Mount static files
app.mount("/built", StaticFiles(directory="static/built"), name="built")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    try:
        logger.debug("Attempting to serve index.html")
        if os.path.exists('static/index.html'):
            logger.debug("Found static/index.html")
            return FileResponse('static/index.html')
        else:
            logger.error("static/index.html not found")
            return {"error": "Frontend not found"}
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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


@app.get("/api/movies/discover")
async def discover_movies(page: int = 1):
    try:
        response = requests.get(
            "https://api.themoviedb.org/3/discover/movie",
            params={
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": page,
                "sort_by": "popularity.desc",
            },
        )
        if response.ok:
            return response.json()
        raise HTTPException(status_code=response.status_code)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/movies/{movie_id}")
async def get_movie_details(movie_id: str):
    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
        )
        if response.ok:
            return response.json()
        raise HTTPException(status_code=response.status_code)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
