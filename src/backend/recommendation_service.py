import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import Dict, List, Tuple, Set, Optional
import logging
import os
import pickle
import json
from datetime import datetime
import requests
import time
from functools import lru_cache


class MovieRecommender:
    """
    A collaborative filtering-based movie recommendation system that uses both user-user similarity
    and item-item similarity to generate personalized movie recommendations.
    """

    def __init__(self, ratings_file: str):
        """
        Initialize the MovieRecommender with a ratings dataset.

        Args:
            ratings_file (str): Path to the Parquet file containing movie ratings
                              Expected format: userId,tmdbId,rating
        """
        self.logger = logging.getLogger(__name__)
        self.model_dir = "model_cache"
        self.ratings_file = ratings_file
        self.api_cache_file = os.path.join(self.model_dir, "api_cache.json")
        self.api_cache = {}
        
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Load API cache if it exists
            if os.path.exists(self.api_cache_file):
                with open(self.api_cache_file, 'r') as f:
                    self.api_cache = json.load(f)
            
            if self.should_train_new_model():
                self.logger.info("Training new model...")
                self.train_model()
                self.save_model()
            else:
                self.logger.info("Loading cached model...")
                self.load_model()
                
            self.log_data_statistics()
            self.logger.info("MovieRecommender initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing MovieRecommender: {str(e)}")
            raise e

    def should_train_new_model(self) -> bool:
        """
        Check if we need to train a new model based on data changes
        and cache freshness.
        """
        metadata_file = os.path.join(self.model_dir, "metadata.json")
        
        # If no metadata exists, we need to train
        if not os.path.exists(metadata_file):
            return True
            
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if ratings file has been modified
            ratings_mtime = os.path.getmtime(self.ratings_file)
            last_trained = metadata.get('last_trained', 0)
            
            # Train if data is newer than model
            if ratings_mtime > last_trained:
                self.logger.info("Ratings data has been updated, retraining model...")
                return True
                
            # Check if all model files exist
            required_files = [
                "normalized_ratings.npz",
                "user_movie_matrix.npz",
                "movie_user_matrix.npz",
                "movie_popularity.npy",
                "top_similar_movies.pkl",
                "mappings.pkl"
            ]
            
            for file in required_files:
                if not os.path.exists(os.path.join(self.model_dir, file)):
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking model freshness: {str(e)}")
            return True

    def save_model(self):
        """Save the trained model and its components."""
        try:
            # Save sparse matrices
            save_npz(os.path.join(self.model_dir, "normalized_ratings.npz"), 
                    self.normalized_ratings)
            save_npz(os.path.join(self.model_dir, "user_movie_matrix.npz"), 
                    self.user_movie_matrix)
            save_npz(os.path.join(self.model_dir, "movie_user_matrix.npz"), 
                    self.movie_user_matrix)
            
            # Save movie popularity
            np.save(os.path.join(self.model_dir, "movie_popularity.npy"), self.movie_popularity)
            
            # Save pre-calculated similarities
            with open(os.path.join(self.model_dir, "top_similar_movies.pkl"), 'wb') as f:
                pickle.dump(self.top_similar_movies, f)
            
            # Save API cache
            with open(self.api_cache_file, 'w') as f:
                json.dump(self.api_cache, f)
            
            # Save mappings and other data
            mappings = {
                'user_id_map': self.user_id_map,
                'movie_id_map': self.movie_id_map,
                'reverse_movie_id_map': self.reverse_movie_id_map,
                'user_means': self.user_means.to_dict(as_series=False)
            }
            
            with open(os.path.join(self.model_dir, "mappings.pkl"), 'wb') as f:
                pickle.dump(mappings, f)
            
            # Save metadata
            metadata = {
                'last_trained': datetime.now().timestamp(),
                'ratings_file': self.ratings_file,
                'n_users': len(self.user_id_map),
                'n_movies': len(self.movie_id_map)
            }
            
            with open(os.path.join(self.model_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f)
                
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise e

    def load_model(self):
        """Load the trained model and its components."""
        try:
            # Load sparse matrices
            self.normalized_ratings = load_npz(
                os.path.join(self.model_dir, "normalized_ratings.npz")
            )
            self.user_movie_matrix = load_npz(
                os.path.join(self.model_dir, "user_movie_matrix.npz")
            )
            self.movie_user_matrix = load_npz(
                os.path.join(self.model_dir, "movie_user_matrix.npz")
            )
            
            # Load movie popularity
            self.movie_popularity = np.load(os.path.join(self.model_dir, "movie_popularity.npy"))
            
            # Load pre-calculated similarities
            with open(os.path.join(self.model_dir, "top_similar_movies.pkl"), 'rb') as f:
                self.top_similar_movies = pickle.load(f)
            
            # Load mappings and other data
            with open(os.path.join(self.model_dir, "mappings.pkl"), 'rb') as f:
                mappings = pickle.load(f)
                
            self.user_id_map = mappings['user_id_map']
            self.movie_id_map = mappings['movie_id_map']
            self.reverse_movie_id_map = mappings['reverse_movie_id_map']
            
            # Load ratings data
            self.ratings_pl = pl.read_parquet(self.ratings_file)
            
            # Reconstruct movie_map and user_map from the mappings
            self.movie_map = pl.DataFrame({
                "tmdbId": list(self.movie_id_map.keys()),
                "movie_idx": list(self.movie_id_map.values())
            })
            
            self.user_map = pl.DataFrame({
                "userId": list(self.user_id_map.keys()),
                "user_idx": list(self.user_id_map.values())
            })
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise e

    def train_model(self):
        """
        Train the recommendation model using Polars for efficient data processing.
        Builds both user-based and item-based similarity matrices.
        """
        # Step 1: Load and process ratings data efficiently with Polars
        self.logger.info("Loading ratings data...")
        
        # Load ratings data directly instead of using scan_parquet
        self.ratings_pl = pl.read_parquet(self.ratings_file).filter(
            pl.col("userId").is_not_null() & 
            pl.col("tmdbId").is_not_null() & 
            pl.col("rating").is_not_null()
        ).with_columns([
            pl.col("tmdbId").cast(pl.Int64),
            pl.col("userId").cast(pl.Int64)
        ])
        
        self.logger.info("Creating user and movie mappings...")
        
        # Create mappings directly
        self.user_map = (self.ratings_pl
            .select("userId")
            .unique()
            .sort("userId")
            .with_row_count("user_idx")
        )
        
        self.movie_map = (self.ratings_pl
            .select("tmdbId")
            .unique()
            .sort("tmdbId")
            .with_row_count("movie_idx")
        )
        
        # Debug logging for movie IDs
        self.logger.debug(f"Number of unique movies: {len(self.movie_map)}")
        self.logger.debug(f"Sample movie IDs: {self.movie_map['tmdbId'].head(20).to_list()}")
        
        # Create efficient lookup dictionaries
        self.user_id_map = dict(zip(self.user_map["userId"], self.user_map["user_idx"]))
        self.movie_id_map = dict(zip(self.movie_map["tmdbId"], self.movie_map["movie_idx"]))
        self.reverse_movie_id_map = dict(zip(self.movie_map["movie_idx"], self.movie_map["tmdbId"]))
        
        self.logger.info("Calculating user means...")
        
        # Calculate user means
        self.user_means = (self.ratings_pl
            .groupby("userId")
            .agg(pl.col("rating").mean())
            .sort("userId")
        )
        
        self.logger.info("Creating sparse matrix...")
        
        # Create sparse matrix
        user_indices = [self.user_id_map[uid] for uid in self.ratings_pl["userId"]]
        movie_indices = [self.movie_id_map[mid] for mid in self.ratings_pl["tmdbId"]]
        ratings = self.ratings_pl["rating"].to_numpy()
        
        self.user_movie_matrix = csr_matrix(
            (ratings, (user_indices, movie_indices)),
            shape=(len(self.user_id_map), len(self.movie_id_map))
        )
        
        self.logger.info("Pre-calculating normalized ratings...")
        
        # Vectorized normalization for all users at once
        user_means_array = np.array([
            self.user_means.filter(pl.col("userId") == uid)["rating"].item()
            for uid in self.user_id_map.keys()
        ])
        
        # Pre-calculate all normalized ratings at once
        normalized_matrix = self.user_movie_matrix.copy()
        for i in range(normalized_matrix.shape[0]):
            row = normalized_matrix[i]
            row_mean = user_means_array[i]
            # Only normalize non-zero elements
            row.data -= row_mean
        
        self.normalized_ratings = normalized_matrix
        
        # Create movie-user matrix for item-based recommendations
        self.logger.info("Creating movie-user matrix for item-based recommendations...")
        self.movie_user_matrix = self.user_movie_matrix.transpose().tocsr()
        
        # Calculate movie popularity (number of ratings per movie)
        self.movie_popularity = np.array(self.movie_user_matrix.getnnz(axis=1)).flatten()
        
        # Pre-calculate top similar movies for most popular movies to speed up recommendations
        self.logger.info("Pre-calculating similarities for top movies...")
        
        # Get top 2000 most popular movies
        top_n = 2000
        top_movie_indices = np.argsort(self.movie_popularity)[-top_n:]
        
        # Create a dictionary to store top similar movies for each popular movie
        self.top_similar_movies = {}
        
        # For each popular movie, find its top 50 similar movies
        for idx, movie_idx in enumerate(top_movie_indices):
            if idx % 100 == 0:
                self.logger.info(f"Processed {idx}/{len(top_movie_indices)} popular movies")
                
            movie_vector = self.movie_user_matrix[movie_idx]
            
            # Calculate similarities with all other movies
            similarities = []
            
            for other_idx in top_movie_indices:
                if other_idx != movie_idx:
                    other_vector = self.movie_user_matrix[other_idx]
                    
                    # Calculate cosine similarity
                    norm_a = np.sqrt(movie_vector.power(2).sum())
                    norm_b = np.sqrt(other_vector.power(2).sum())
                    
                    if norm_a > 0 and norm_b > 0:
                        dot_product = movie_vector.dot(other_vector.transpose())[0, 0]
                        similarity = dot_product / (norm_a * norm_b)
                        
                        if similarity > 0:
                            similarities.append((other_idx, similarity))
            
            # Sort by similarity and keep top 50
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.top_similar_movies[movie_idx] = similarities[:50]
        
        self.logger.info("Model training completed")

    def get_popular_movies(self, n: int = 20) -> Dict[int, int]:
        """
        Get the most viewed movies.

        Args:
            n (int): Number of popular movies to return

        Returns:
            Dict[int, int]: Dictionary of {tmdb_id: view_count}
        """
        # Use the already loaded ratings data
        movie_views = self.ratings_pl.groupby("tmdbId").agg(
            pl.count().alias("count")
        ).sort("count", descending=True)
        
        # Get top N movies
        top_movies = movie_views.limit(n)

        # Convert to dictionary
        popular_movies = {
            int(row["tmdbId"]): int(row["count"]) 
            for row in top_movies.iter_rows(named=True)
        }

        # Add debug logging
        self.logger.debug(f"Found {len(popular_movies)} popular movies")
        self.logger.debug(f"Sample of popular movies: {dict(list(popular_movies.items())[:5])}")

        return popular_movies

    def get_recommendations(
        self, user_preferences: Dict[int, float], n_recommendations: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate recommendations using both user-based and item-based collaborative filtering.
        Also handles movies not found in the dataset by using API recommendations.
        """
        start_time = time.time()
        
        try:
            # Track movies not found in our dataset
            missing_movies = []
            
            # Initial setup and validation
            temp_user = np.zeros(len(self.movie_id_map))
            valid_ratings = []
            rated_movies = set()
            
            # Log input stats
            self.logger.debug(f"Processing {len(user_preferences)} ratings")
            
            for movie_id, rating in user_preferences.items():
                if movie_id in self.movie_id_map:
                    idx = self.movie_id_map[movie_id]
                    temp_user[idx] = rating
                    valid_ratings.append(rating)
                    rated_movies.add(movie_id)
                else:
                    # Track movies not in our dataset for API fallback
                    missing_movies.append(movie_id)
            
            # Initialize recommendations dictionary
            recommendations = {}
            
            # If we have valid ratings in our dataset, use collaborative filtering
            if valid_ratings:
                # Log normalization stats
                user_mean = np.mean(valid_ratings)
                user_std = np.std(valid_ratings)
                global_mean = float(self.ratings_pl["rating"].mean())
                global_std = float(self.ratings_pl["rating"].std())
                self.logger.debug(f"User stats - mean: {user_mean:.2f}, std: {user_std:.2f}")
                self.logger.debug(f"Global stats - mean: {global_mean:.2f}, std: {global_std:.2f}")
                
                # Normalize ratings
                mask = temp_user != 0
                if user_std > 0:
                    temp_user[mask] = 0.7 * ((temp_user[mask] - user_mean) / user_std) + \
                                    0.3 * ((temp_user[mask] - global_mean) / global_std)
                else:
                    temp_user[mask] = (temp_user[mask] - global_mean) / global_std
                
                # Log normalized ratings
                self.logger.debug(f"Normalized ratings range: {temp_user[mask].min():.2f} to {temp_user[mask].max():.2f}")
                
                # PART 1: USER-BASED COLLABORATIVE FILTERING
                # Calculate similarities
                similarities = cosine_similarity(temp_user.reshape(1, -1), self.normalized_ratings)[0]
                
                # Get top similar users
                top_n = 50
                top_user_indices = np.argsort(similarities)[-top_n:]
                top_similarities = similarities[top_user_indices]
                
                # Log similarity stats
                self.logger.debug(f"User similarity range: {similarities.min():.2f} to {similarities.max():.2f}")
                self.logger.debug(f"Top 5 user similarities: {top_similarities[-5:]}")
                
                # Generate predictions
                weighted_sum = np.zeros(self.user_movie_matrix.shape[1])
                similarity_sum = np.zeros(self.user_movie_matrix.shape[1])
                
                for idx, sim in zip(top_user_indices, top_similarities):
                    if sim > 0:
                        user_ratings = self.user_movie_matrix[idx].toarray().flatten()
                        weight = np.power(sim, 2)
                        weighted_sum += weight * user_ratings
                        similarity_sum += weight * (user_ratings != 0)
                
                # Calculate predicted ratings
                mask = similarity_sum > 0
                user_based_predictions = np.zeros_like(weighted_sum)
                user_based_predictions[mask] = weighted_sum[mask] / similarity_sum[mask]
                
                # PART 2: ITEM-BASED COLLABORATIVE FILTERING - OPTIMIZED VERSION
                item_based_predictions = np.zeros(len(self.movie_id_map))
                
                # Get indices of rated movies
                rated_indices = [self.movie_id_map[movie_id] for movie_id in rated_movies if movie_id in self.movie_id_map]
                
                # Use pre-calculated similarities for faster recommendations
                if rated_indices:
                    # Find which rated movies are in our pre-calculated set
                    precalc_rated = [idx for idx in rated_indices if idx in self.top_similar_movies]
                    
                    # For each rated movie that has pre-calculated similarities
                    for rated_idx in precalc_rated:
                        # Get the rating for this movie
                        rating = temp_user[rated_idx]
                        
                        # Get its similar movies
                        for similar_idx, similarity in self.top_similar_movies[rated_idx]:
                            # Skip if already rated
                            if similar_idx in rated_indices:
                                continue
                                
                            # Add weighted rating to prediction
                            if item_based_predictions[similar_idx] == 0:
                                item_based_predictions[similar_idx] = rating * similarity
                            else:
                                # Average with existing prediction
                                item_based_predictions[similar_idx] = (item_based_predictions[similar_idx] + rating * similarity) / 2
                
                # PART 3: COMBINE PREDICTIONS
                # Combine user-based and item-based predictions with a weighted average
                # We'll use a 60-40 split favoring item-based for more relevant recommendations
                combined_predictions = np.zeros_like(user_based_predictions)
                
                # Where both predictions exist
                both_mask = (user_based_predictions > 0) & (item_based_predictions > 0)
                combined_predictions[both_mask] = 0.4 * user_based_predictions[both_mask] + 0.6 * item_based_predictions[both_mask]
                
                # Where only user-based exists
                user_only_mask = (user_based_predictions > 0) & (item_based_predictions == 0)
                combined_predictions[user_only_mask] = user_based_predictions[user_only_mask]
                
                # Where only item-based exists
                item_only_mask = (user_based_predictions == 0) & (item_based_predictions > 0)
                combined_predictions[item_only_mask] = item_based_predictions[item_only_mask]
                
                # Scale predictions while preserving relative differences
                nonzero_mask = combined_predictions > 0
                if nonzero_mask.any():
                    # Get prediction statistics
                    pred_min = combined_predictions[nonzero_mask].min()
                    pred_max = combined_predictions[nonzero_mask].max()
                    pred_mean = combined_predictions[nonzero_mask].mean()
                    pred_std = combined_predictions[nonzero_mask].std()
                    
                    if pred_std > 0:
                        # Use z-score transformation then scale to target range
                        z_scores = (combined_predictions[nonzero_mask] - pred_mean) / pred_std
                        # Scale to 3.5-5.0 range with sigmoid-like squashing
                        scaled_ratings = 3.5 + 1.5 / (1 + np.exp(-z_scores))
                        combined_predictions[nonzero_mask] = scaled_ratings
                    else:
                        # If no variation, scale linearly
                        combined_predictions[nonzero_mask] = 3.5 + 1.5 * (
                            (combined_predictions[nonzero_mask] - pred_min) / 
                            (pred_max - pred_min if pred_max > pred_min else 1)
                        )
                
                # Convert to recommendations
                for idx, rating in enumerate(combined_predictions):
                    if rating > 3.5:
                        movie_id = self.reverse_movie_id_map[idx]
                        if movie_id not in rated_movies:
                            recommendations[str(movie_id)] = round(float(rating), 2)
                
                # Log final recommendations stats
                self.logger.debug(f"Generated {len(recommendations)} recommendations above threshold")
                if recommendations:
                    ratings_list = list(recommendations.values())
                    self.logger.debug(f"Final ratings range: {min(ratings_list):.2f} to {max(ratings_list):.2f}")
            
            # PART 4: API RECOMMENDATIONS FOR MISSING MOVIES
            # If we have missing movies or not enough recommendations, use API
            api_recommendations = {}
            
            if missing_movies or len(recommendations) < n_recommendations:
                # Get API recommendations for missing movies
                api_recommendations = self.get_api_recommendations(
                    missing_movies, 
                    rated_movies,
                    n_recommendations - len(recommendations)
                )
                
                # Add API recommendations to our recommendations
                recommendations.update(api_recommendations)
            
            # Log timing
            end_time = time.time()
            self.logger.info(f"Recommendation generation took {end_time - start_time:.2f} seconds")
            
            # Return the combined recommendations
            return {
                'recommendations': dict(sorted(
                    recommendations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:n_recommendations])
            }
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {'recommendations': {}, 'error': str(e)}
    
    def get_api_recommendations(
        self, 
        missing_movies: List[int], 
        rated_movies: Set[int],
        n_needed: int
    ) -> Dict[str, float]:
        """
        Get recommendations from TMDB API for movies not in our dataset.
        
        Args:
            missing_movies: List of movie IDs not found in our dataset
            rated_movies: Set of movie IDs already rated by the user
            n_needed: Number of recommendations needed
            
        Returns:
            Dictionary of movie IDs to predicted ratings
        """
        if not missing_movies or n_needed <= 0:
            return {}
            
        try:
            # TMDB API configuration
            api_key = os.environ.get("TMDB_API_KEY", "")
            if not api_key:
                self.logger.warning("No TMDB API key found, skipping API recommendations")
                return {}
                
            base_url = "https://api.themoviedb.org/3"
            
            # Get similar movies for each missing movie
            similar_movies = {}
            
            # Set a timeout for API calls to avoid long waits
            api_timeout = 2.0  # seconds
            
            for movie_id in missing_movies:
                # Skip if we already have enough recommendations
                if len(similar_movies) >= n_needed:
                    break
                
                # Check if we have cached results for this movie
                cache_key = str(movie_id)
                if cache_key in self.api_cache:
                    self.logger.info(f"Using cached API results for movie {movie_id}")
                    cached_results = self.api_cache[cache_key]
                    
                    # Filter out movies already rated
                    filtered_results = {
                        tmdb_id: rating for tmdb_id, rating in cached_results.items()
                        if int(tmdb_id) not in rated_movies
                    }
                    
                    # Add to similar movies
                    similar_movies.update(filtered_results)
                    continue
                
                # Start API call timer
                api_start_time = time.time()
                
                # Get movie details to get the title for logging
                try:
                    movie_url = f"{base_url}/movie/{movie_id}?api_key={api_key}"
                    movie_response = requests.get(movie_url, timeout=api_timeout)
                    
                    if movie_response.status_code == 200:
                        movie_data = movie_response.json()
                        movie_title = movie_data.get("title", f"Movie {movie_id}")
                        self.logger.info(f"Fetching similar movies for: {movie_title} ({movie_data.get('release_date', '')[:4]})")
                        
                        # Get similar movies
                        similar_url = f"{base_url}/movie/{movie_id}/similar?api_key={api_key}"
                        similar_response = requests.get(similar_url, timeout=api_timeout)
                        
                        if similar_response.status_code == 200:
                            similar_data = similar_response.json()
                            results = similar_data.get("results", [])
                            
                            # Filter out movies already rated
                            filtered_results = [
                                movie for movie in results 
                                if movie.get("id") not in rated_movies
                            ]
                            
                            # Cache the results
                            movie_cache = {}
                            
                            # Add to similar movies with a high rating (these are API recommendations)
                            for movie in filtered_results:
                                tmdb_id = movie.get("id")
                                # Use vote_average as a basis for our rating, scaled to our 1-5 range
                                vote_average = movie.get("vote_average", 5.0)
                                # Scale from TMDB's 0-10 to our 1-5
                                scaled_rating = 1.0 + (vote_average / 10.0) * 4.0
                                # Boost a bit since these are similar to movies the user liked
                                boosted_rating = min(5.0, scaled_rating * 1.1)
                                
                                # Add to results and cache
                                similar_movies[str(tmdb_id)] = round(boosted_rating, 2)
                                movie_cache[str(tmdb_id)] = round(boosted_rating, 2)
                            
                            # Update cache
                            self.api_cache[cache_key] = movie_cache
                            
                            # Save cache periodically
                            with open(self.api_cache_file, 'w') as f:
                                json.dump(self.api_cache, f)
                            
                            self.logger.info(f"Found {len(filtered_results)} similar movies for {movie_title}")
                        else:
                            self.logger.warning(f"Failed to get similar movies for {movie_id}: {similar_response.status_code}")
                    else:
                        self.logger.warning(f"Failed to get movie details for {movie_id}: {movie_response.status_code}")
                
                except requests.exceptions.Timeout:
                    self.logger.warning(f"API request timed out for movie {movie_id}")
                except Exception as e:
                    self.logger.warning(f"Error in API request for movie {movie_id}: {str(e)}")
                
                # Check if we've spent too much time on API calls
                api_elapsed = time.time() - api_start_time
                if api_elapsed > 5.0:  # If a single movie takes more than 5 seconds, stop API calls
                    self.logger.warning(f"API calls taking too long ({api_elapsed:.2f}s), skipping remaining movies")
                    break
            
            # Return the similar movies as recommendations
            self.logger.info(f"Found {len(similar_movies)} similar movies across {len(missing_movies)} input movies")
            return similar_movies
            
        except Exception as e:
            self.logger.error(f"Error getting API recommendations: {str(e)}")
            return {}

    def log_data_statistics(self):
        """
        Log important statistics about the loaded dataset for monitoring and debugging.
        """
        # Make sure we have ratings data
        if not hasattr(self, 'ratings_pl'):
            self.ratings_pl = pl.read_parquet(self.ratings_file)
        
        self.logger.info(f"Total number of ratings: {len(self.ratings_pl)}")
        self.logger.info(f"Number of unique movies: {len(self.movie_id_map)}")
        self.logger.info(f"Number of unique users: {len(self.user_id_map)}")
        
        # Get movie ID range
        min_movie_id = self.movie_map["tmdbId"].min()
        max_movie_id = self.movie_map["tmdbId"].max()
        self.logger.info(f"Movie ID range: {min_movie_id} to {max_movie_id}")
        
        # Rating statistics
        rating_stats = self.ratings_pl.select(
            pl.col("rating").mean().alias("mean"),
            pl.col("rating").std().alias("std"),
            pl.col("rating").min().alias("min"),
            pl.col("rating").max().alias("max"),
            pl.col("rating").median().alias("median")
        )
        self.logger.info(f"Rating statistics:\n{rating_stats}")
