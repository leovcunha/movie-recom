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
import gc  # Add garbage collection
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
import numpy as np

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
    # Debug logging
    logger.debug(f"Current working directory: {os.getcwd()}")
    
    # Use the static directory where Dockerfile copied files
    static_path = "static"
    
    logger.debug(f"Static path: {static_path}")
    logger.debug(f"Static directory exists: {os.path.exists(static_path)}")
    if os.path.exists(static_path):
        logger.debug(f"Static directory contents: {os.listdir(static_path)}")
    
    # Mount static files
    app.mount("/built", StaticFiles(directory=os.path.join(static_path, "built")), name="built")
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Add at the start of your file, after creating the FastAPI app
app.movie_cache = {}

# Add this function to prefetch genre data
async def prefetch_genre_data():
    # Limit to just 3 genres to save memory
    genres = [28, 35, 18]  # Action, Comedy, Drama
    timeout = aiohttp.ClientTimeout(total=10)
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
                    
                    # Only take first 10 movies and just essential fields to save memory
                    results = []
                    for movie in data.get('results', [])[:10]:
                        results.append({
                            'id': movie['id'],
                            'title': movie['title'],
                            'poster_path': movie.get('poster_path', ''),
                            'vote_average': movie.get('vote_average', 0)
                        })
                    
                    app.movie_cache[cache_key] = {
                        'results': results,
                        'page': data['page'],
                        'total_pages': data['total_pages'],
                        'total_results': data['total_results']
                    }
                    
                    # Force garbage collection after processing each genre
                    gc.collect()
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
            
            # Direct path to index.html based on Dockerfile
            index_path = "static/index.html"
            
            logger.debug(f"Index path: {index_path}")
            logger.debug(f"Index file exists: {os.path.exists(index_path)}")
            
            return FileResponse(index_path)
        else:
            return {"message": "API running in development mode"}
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update path to be relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data", "ratings_tmdb.parquet")  # Changed to parquet file

# Initialize recommender once when starting the server
try:
    # Force garbage collection before loading
    gc.collect()
    
    # Monitor memory if available
    if HAVE_PSUTIL:
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        mem_available = psutil.virtual_memory().available / (1024 * 1024)
        logger.info(f"Memory before loading recommender: {mem_before:.2f} MB, Available: {mem_available:.2f} MB")
        
        # Set environment variable for extreme memory optimization if needed
        if mem_available < 100:  # Less than 100MB available
            os.environ["EXTREME_MEMORY_OPTIMIZATION"] = "1"
            logger.warning("Setting EXTREME_MEMORY_OPTIMIZATION=1 due to low available memory")
    
    # Initialize recommender
    logger.info("Loading movie recommender...")
    recommender = MovieRecommender(data_path)
    
    # Force garbage collection after loading model to free up memory
    gc.collect()
    
    # Log memory after loading
    if HAVE_PSUTIL:
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info(f"Memory after loading recommender: {mem_after:.2f} MB")
    
    logger.info("Recommender loaded successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommender: {e}")
    raise


class UserPreferences(BaseModel):
    ratings: Dict[int, float]


@app.post("/api/recommendations")
async def get_recommendations(preferences: UserPreferences):
    try:
        # Debug logging
        logger.debug(f"Received preferences: {preferences.ratings}")
        
        # Get recommendations from the model
        recommendations = recommender.get_recommendations(preferences.ratings)
        logger.debug(f"Raw recommendations from model: {recommendations}")
        
        # Force garbage collection after getting recommendations
        gc.collect()
            
        # Check if movies exist in our dataset more accurately
        missing_movie_ids = []
        movie_ids_in_dataset = set(recommender.movie_map["tmdbId"].to_list())
        
        for movie_id in preferences.ratings:
            # Debug logging for movie existence check
            logger.debug(f"Checking movie {movie_id} in dataset")
            
            if movie_id not in movie_ids_in_dataset and preferences.ratings[movie_id] > 3.0:
                missing_movie_ids.append(movie_id)
                logger.debug(f"Movie {movie_id} not found in dataset")
                
        if missing_movie_ids:
            logger.info(f"Movies not found in local data: {missing_movie_ids}")
            
            # Fetch similar movies for each missing movie
            similar_movies = await fetch_similar_movies_from_tmdb(missing_movie_ids)
            logger.debug(f"Similar movies fetched: {similar_movies}")
            
            # Add similar movies to recommendations
            for movie_id, similar_movie_list in similar_movies.items():
                for similar_movie in similar_movie_list:
                    if 'recommendations' not in recommendations:
                        recommendations['recommendations'] = {}
                    recommendations['recommendations'][str(similar_movie['id'])] = round(similar_movie['vote_average'] / 2, 1)
            
        if not recommendations or not recommendations.get('recommendations'):
            logger.warning("No recommendations generated")
            raise HTTPException(status_code=404, detail="No recommendations found")

        # Debug logging before fetching movie details
        logger.debug(f"Final recommendations before details: {recommendations}")

        # Fetch movie details for each recommendation
        recommended_movies = []
        
        # Process in smaller batches to prevent memory issues
        batch_size = 10
        all_items = list(recommendations['recommendations'].items())
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(all_items), batch_size):
                batch = all_items[i:i+batch_size]
                
                tasks = []
                for movie_id, predicted_rating in batch:
                    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                    params = {
                        "api_key": TMDB_API_KEY,
                        "language": "en-US"
                    }
                    tasks.append(session.get(url, params=params))
                
                responses = await asyncio.gather(*tasks)
                
                for response, (movie_id, predicted_rating) in zip(responses, batch):
                    if response.status == 200:
                        movie_data = await response.json()
                        recommended_movies.append({
                            'id': int(movie_id),
                            'title': movie_data['title'],
                            'overview': movie_data['overview'],
                            'poster_path': movie_data['poster_path'],
                            'backdrop_path': movie_data['backdrop_path'],
                            'predicted_rating': predicted_rating,
                            'userRating': preferences.ratings.get(int(movie_id), 0)
                        })
                
                # Force garbage collection after each batch
                gc.collect()

        # Sort by predicted rating
        recommended_movies.sort(key=lambda x: x['predicted_rating'], reverse=True)

        # Force final garbage collection
        gc.collect()
        
        return recommended_movies

    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_similar_movies_from_tmdb(movie_ids: list[int], n: int = 3) -> Dict[int, list]:
    """
    Fetches similar movies from TMDB for a list of movie IDs, filtered to be within ±3 years 
    of the original movie's release year and with a minimum rating of 3.5.
    Optimized for lower memory usage.

    Args:
        movie_ids (list[int]): List of movie IDs for which to find similar movies.
        n (int): Number of similar movies to return for each movie.

    Returns:
        Dict[int, list]: Dictionary of {movie_id: list of similar movies}
    """
    # Process at most 3 movies to save memory
    movie_ids = movie_ids[:3]
    
    similar_movies_dict = {}
    async with aiohttp.ClientSession() as session:
        for movie_id in movie_ids:
            # First get the original movie's release year
            try:
                url = f"https://api.themoviedb.org/3/movie/{movie_id}"
                params = {
                    "api_key": TMDB_API_KEY,
                    "language": "en-US"
                }
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        movie_data = await response.json()
                        release_date = movie_data.get('release_date', '')
                        
                        if release_date:
                            release_year = int(release_date[:4])
                            min_year = release_year - 3
                            max_year = release_year + 3
                            
                            # Now get similar movies
                            url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar"
                            params = {
                                "api_key": TMDB_API_KEY,
                                "language": "en-US",
                                "page": "1"
                            }
                            async with session.get(url, params=params) as similar_response:
                                if similar_response.status == 200:
                                    data = await similar_response.json()
                                    
                                    # Filter similar movies efficiently with list comprehension
                                    similar_movies = []
                                    for movie in data.get('results', [])[:10]:  # Only check first 10 movies
                                        if (movie.get('release_date', '') and 
                                            min_year <= int(movie['release_date'][:4]) <= max_year and
                                            movie.get('vote_average', 0)/2 >= 3.5):
                                            # Only include essential fields
                                            similar_movies.append({
                                                'id': movie['id'],
                                                'title': movie['title'],
                                                'vote_average': movie['vote_average'],
                                                'release_date': movie['release_date'][:4]
                                            })
                                            
                                            # Break early if we have enough movies
                                            if len(similar_movies) >= n:
                                                break
                                    
                                    similar_movies_dict[movie_id] = similar_movies
                                else:
                                    similar_movies_dict[movie_id] = []
                        else:
                            similar_movies_dict[movie_id] = []
                    else:
                        similar_movies_dict[movie_id] = []
            except Exception as e:
                logger.error(f"Error fetching similar movies for {movie_id}: {e}")
                similar_movies_dict[movie_id] = []
                
            # Force garbage collection after processing each movie
            gc.collect()
    
    return similar_movies_dict


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

        # If not in cache, create a minimal response
        all_movie_ids = recommender.movie_ids[:50]  # Just get first 50 IDs instead of all
        
        # Get popular movies but limit to just 10
        popular_movies = recommender.get_popular_movies(10)

        # Process only a few movies to save memory
        movie_sample = list(popular_movies.keys())[:5]  # Just 5 movies for sample
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for movie_id in movie_sample:
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
                    # Only include essential fields
                    movies_data.append({
                        'id': movie_data['id'],
                        'title': movie_data['title'],
                        'poster_path': movie_data['poster_path'],
                        'vote_average': movie_data['vote_average']
                    })

        result = {
            "total_movies": len(recommender.movie_ids),
            "id_range": f"{min(recommender.movie_ids)} to {max(recommender.movie_ids)}",
            "popular_movies": {int(k): int(v) for k, v in list(popular_movies.items())[:10]},
            "sample_ids": [int(id) for id in all_movie_ids],
            "movie_details": movies_data
        }

        # Cache the result
        app.valid_movies_cache = result
        
        # Force garbage collection
        gc.collect()
        
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
            
            # If it's Action genre, include Adventure genre as well
            genres_param = f"{genre_id},12" if genre_id == 28 else str(genre_id)
            
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "with_genres": genres_param,
                "sort_by": "popularity.desc",
                "include_adult": "false",
                "include_video": "false",
                "page": str(page),
                "vote_count.gte": "200",
                "vote_average.gte": "6.0",
                "with_original_language": "en"
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
async def get_popular_movies(page: int = 1):
    try:
        # Get popular movies from recommender with pagination
        items_per_page = 20
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        
        # Get all popular movies and slice for pagination
        all_popular_movies = recommender.get_popular_movies(100)  # Get more movies to support pagination
        movie_ids = list(all_popular_movies.keys())[start_idx:end_idx]
        
        if not movie_ids:
            return {
                "results": [],
                "total_results": len(all_popular_movies),
                "page": page,
                "total_pages": (len(all_popular_movies) + items_per_page - 1) // items_per_page
            }

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
            for response, movie_id in zip(responses, movie_ids):
                if response.status == 200:
                    movie_data = await response.json()
                    # Add view count from our dataset
                    movie_data['view_count'] = all_popular_movies[movie_id]
                    movies_data.append(movie_data)

        total_movies = len(all_popular_movies)
        total_pages = (total_movies + items_per_page - 1) // items_per_page

        return {
            "results": movies_data,
            "total_results": total_movies,
            "page": page,
            "total_pages": total_pages
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/movies/details/{movie_id}")
async def get_movie_details(movie_id: int):
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    movie_data = await response.json()
                    return {
                        'id': movie_data['id'],
                        'title': movie_data['title'],
                        'overview': movie_data['overview'],
                        'poster_path': movie_data['poster_path'],
                        'backdrop_path': movie_data['backdrop_path'],
                        'vote_average': movie_data['vote_average'],
                        'release_date': movie_data['release_date']
                    }
                raise HTTPException(status_code=response.status, detail=f"TMDB API error: {response.status}")
    except Exception as e:
        logger.error(f"Error getting movie details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/movies/{movie_id}/videos")
async def get_movie_videos(movie_id: int):
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Filter for YouTube trailers (most common)
                    trailers = [
                        video for video in data.get('results', [])
                        if video.get('site') == 'YouTube' and 
                        (video.get('type') == 'Trailer' or video.get('type') == 'Teaser')
                    ]
                    
                    # Return the first trailer, or an empty object if none found
                    return {
                        'results': trailers,
                        'success': True
                    }
                    
                raise HTTPException(status_code=response.status, detail=f"{response.status}: {response.reason}")
    except Exception as e:
        logger.error(f"Error getting movie videos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add at the top of the file with other constants
GENRES = [
    28,     # Action & Adventure (will fetch both)
    35,     # Comedy
    18,     # Drama
    10751,  # Family
    10749   # Romance
]


@app.get("/api/movies/search")
async def search_movies(query: str, page: int = 1):
    try:
        if not query or len(query.strip()) < 2:
            return {"results": [], "page": 1, "total_pages": 0, "total_results": 0}
            
        async with aiohttp.ClientSession() as session:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "language": "en-US",
                "query": query,
                "page": str(page),
                "include_adult": "false"
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
                            'backdrop_path': movie.get('backdrop_path'),
                            'vote_average': movie['vote_average'],
                            'release_date': movie.get('release_date', '')
                        } for movie in data['results'] if movie.get('poster_path')],  # Filter out movies without posters
                        'page': data['page'],
                        'total_pages': data['total_pages'],
                        'total_results': data['total_results']
                    }
                    
                    return result
                
                raise HTTPException(status_code=response.status, detail=f"{response.status}: {response.reason}")
    except Exception as e:
        logger.error(f"Error searching movies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
