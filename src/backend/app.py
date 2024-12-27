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
from datetime import datetime, timedelta
import aiohttp
import asyncio

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
ENV = os.getenv("ENV", "development")  # Default to development if not set

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files only in production
if ENV == "production":
    # Debug: Print current directory and contents
    logger.debug(f"Current working directory: {os.getcwd()}")
    logger.debug(f"Directory contents: {os.listdir('.')}")
    logger.debug(f"Static directory contents: {os.listdir('static') if os.path.exists('static') else 'static directory not found'}")

    # Mount static files
    app.mount("/built", StaticFiles(directory="static/built"), name="built")
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Add at the start of your file, after creating the FastAPI app
app.movie_cache = {}

# Add this function to prefetch genre data
async def prefetch_genre_data():
    genres = [28, 12, 35, 18, 10751, 10749]  # Your genre IDs
    timeout = aiohttp.ClientTimeout(total=10)  # Add timeout
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for genre_id in genres:
                url = "https://api.themoviedb.org/3/discover/movie"
                params = {
                    "api_key": TMDB_API_KEY,
                    "language": "en-US",
                    "with_genres": str(genre_id),
                    "sort_by": "vote_average.desc",
                    "include_adult": "false",
                    "include_video": "false",
                    "page": "1",
                    "vote_count.gte": "100"
                }
                tasks.append(session.get(url, params=params))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for genre_id, response in zip(genres, responses):
                if isinstance(response, Exception):
                    logger.error(f"Error prefetching genre {genre_id}: {response}")
                    continue
                    
                if response.status == 200:
                    data = await response.json()
                    cache_key = f"genre_{genre_id}_page_1"
                    app.movie_cache[cache_key] = {
                        'results': [{
                            'id': movie['id'],
                            'title': movie['title'],
                            'overview': movie['overview'],
                            'poster_path': movie['poster_path'],
                            'backdrop_path': movie['backdrop_path'],
                            'vote_average': movie['vote_average'],
                            'release_date': movie['release_date'],
                            'genre_ids': movie['genre_ids']
                        } for movie in data['results']],
                        'page': data['page'],
                        'total_pages': data['total_pages'],
                        'total_results': data['total_results']
                    }
    except Exception as e:
        logger.error(f"Error in prefetch_genre_data: {e}")
        # Don't let startup fail if prefetch fails
        pass

@app.on_event("startup")
async def startup_event():
    try:
        await prefetch_genre_data()
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Don't let startup fail if prefetch fails
        pass

@app.get("/")
async def read_root():
    try:
        if ENV == "production":
            logger.debug("Attempting to serve index.html")
            return FileResponse('static/index.html')
        else:
            return {"message": "API running in development mode"}
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
        # Check cache first
        if hasattr(app, 'valid_movies_cache'):
            return app.valid_movies_cache

        # If not in cache, fetch and process
        all_movie_ids = [int(id) for id in recommender.movie_ids]
        popular_movies = recommender.get_popular_movies(20)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for movie_id in popular_movies.keys():
                url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                params = {
                    "api_key": TMDB_API_KEY,
                    "language": "en-US"
                }
                tasks.append(session.get(url, params=params))
            
            responses = await asyncio.gather(*tasks)
            movies_data = []
            for response in responses:
                if response.status == 200:
                    movie_data = await response.json()
                    movies_data.append({
                        'id': movie_data['id'],
                        'title': movie_data['title'],
                        'overview': movie_data['overview'],
                        'poster_path': movie_data['poster_path'],
                        'backdrop_path': movie_data['backdrop_path'],
                        'vote_average': movie_data['vote_average'],
                        'release_date': movie_data['release_date'],
                        'genre_ids': movie_data.get('genres', [])
                    })

        result = {
            "total_movies": len(all_movie_ids),
            "id_range": f"{min(all_movie_ids)} to {max(all_movie_ids)}",
            "popular_movies": {int(k): float(v) for k, v in popular_movies.items()},
            "sample_ids": [int(id) for id in all_movie_ids[:50]],
            "movie_details": movies_data
        }

        # Cache the result
        app.valid_movies_cache = result
        return result
    except Exception as e:
        logger.error(f"Error getting valid movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/movies/discover")
async def discover_movies(page: int = 1):
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "page": str(page),
                "sort_by": "popularity.desc",
                "include_adult": "false",
                "include_video": "false"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                raise HTTPException(status_code=response.status, detail=f"{response.status}: {response.reason}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/movies/genres")
async def get_genres():
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://api.themoviedb.org/3/genre/movie/list"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                raise HTTPException(status_code=response.status, detail=f"{response.status}: {response.reason}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/movies/genre/{genre_id}")
async def get_movies_by_genre(genre_id: int, page: int = 1):
    try:
        # Create a cache key
        cache_key = f"genre_{genre_id}_page_{page}"
        if cache_key in app.movie_cache:
            return app.movie_cache[cache_key]

        async with aiohttp.ClientSession() as session:
            url = "https://api.themoviedb.org/3/discover/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "with_genres": str(genre_id),
                "sort_by": "vote_average.desc",
                "include_adult": "false",
                "include_video": "false",
                "page": str(page),
                "vote_count.gte": "100"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    result = {
                        'results': [{
                            'id': movie['id'],
                            'title': movie['title'],
                            'overview': movie['overview'],
                            'poster_path': movie['poster_path'],
                            'backdrop_path': movie['backdrop_path'],
                            'vote_average': movie['vote_average'],
                            'release_date': movie['release_date'],
                            'genre_ids': movie['genre_ids']
                        } for movie in data['results']],
                        'page': data['page'],
                        'total_pages': data['total_pages'],
                        'total_results': data['total_results']
                    }
                    
                    # Cache the result
                    app.movie_cache[cache_key] = result
                    return result
                
                raise HTTPException(status_code=response.status, detail=f"{response.status}: {response.reason}")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/movies/popular")
async def get_popular_movies():
    try:
        # Get popular movies from recommender
        popular_movies = recommender.get_popular_movies(20)
        movie_ids = list(popular_movies.keys())

        # Make parallel requests for movie details
        async with aiohttp.ClientSession() as session:
            tasks = []
            for movie_id in movie_ids:
                url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                params = {
                    "api_key": TMDB_API_KEY,
                    "language": "en-US"
                }
                tasks.append(session.get(url, params=params))
            
            responses = await asyncio.gather(*tasks)
            movies_data = []
            for response in responses:
                if response.status == 200:
                    movie_data = await response.json()
                    movies_data.append(movie_data)

        return {
            "results": movies_data,
            "total_results": len(movies_data),
            "page": 1,
            "total_pages": 1
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
