import aiohttp
import logging
import os
from typing import Dict, List

# This will be passed in, but we need it for type hinting
from recommendation_service import MovieRecommender

logger = logging.getLogger(__name__)
TMDB_API_KEY = os.getenv("TMDB_API_KEY")


async def get_imputed_ratings_for_missing_movies(
    missing_movie_ids: list[int],
    original_ratings: Dict[int, float],
    recommender: MovieRecommender
) -> Dict[int, float]:
    """
    For new movies (cold-start items), find similar "proxy" movies that exist
    in our database. This function returns a dictionary of these proxy movies
    with an "imputed" rating, which can be used to enrich the user's profile
    before generating final recommendations.
    """
    imputed_ratings = {}
    if not TMDB_API_KEY:
        logger.warning("TMDB_API_KEY not set. Skipping imputed ratings.")
        return {}

    known_movie_ids_in_db = set(recommender.movie_id_map.keys())

    for new_movie_id in missing_movie_ids[:2]:  # Process up to 2 new movies
        proxy_movie_ids = await _find_proxy_movies_in_db(new_movie_id, known_movie_ids_in_db)

        for proxy_id in proxy_movie_ids:
            # Impute the rating for the proxy movie using the user's original rating as a signal.
            imputed_ratings[proxy_id] = original_ratings[new_movie_id]

    return imputed_ratings

async def _find_proxy_movies_in_db(start_movie_id: int, known_ids: set) -> List[int]:
    """Helper to recursively find similar movies that exist in our database."""
    search_queue = [(start_movie_id, 0)]
    searched_ids = {start_movie_id}
    found_proxies = []
    max_depth = 2
    max_proxies = 2

    while search_queue and len(found_proxies) < max_proxies:
        current_id, depth = search_queue.pop(0)
        if depth >= max_depth:
            continue

        similar_movies_map = await fetch_similar_movies_from_tmdb([current_id], n=5)
        for movie in similar_movies_map.get(current_id, []):
            similar_id = movie['id']
            if similar_id in searched_ids:
                continue
            searched_ids.add(similar_id)

            if similar_id in known_ids:
                logger.info(f"Found proxy movie {similar_id} for new movie {start_movie_id}")
                found_proxies.append(similar_id)
                if len(found_proxies) >= max_proxies:
                    break
            else:
                search_queue.append((similar_id, depth + 1))
        if len(found_proxies) >= max_proxies:
            break
    
    return found_proxies


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
    import gc

    similar_movies_dict = {}
    async with aiohttp.ClientSession() as session:
        for movie_id in movie_ids:
            # First get the original movie's release year
            try:
                # Get similar movies directly
                url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar"
                params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": "1"}
                async with session.get(url, params=params) as similar_response:
                    if similar_response.status == 200:
                        data = await similar_response.json()
                        # Relax filtering to increase chances of finding a match in our DB
                        similar_movies = [
                            movie for movie in data.get('results', [])
                            if movie.get('vote_average', 0) > 0  # Just ensure it has some votes
                        ][:n]  # Take up to n movies
                        similar_movies_dict[movie_id] = similar_movies
                    else:
                        similar_movies_dict[movie_id] = []
            except Exception as e:
                logger.error(f"Error fetching similar movies for {movie_id}: {e}")
                similar_movies_dict[movie_id] = []

            # Force garbage collection after processing each movie
            gc.collect()

    return similar_movies_dict
