import polars as pl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import Dict
import logging
import os
import pickle
import json
from datetime import datetime


class MovieRecommender:
    """
    A collaborative filtering-based movie recommendation system that uses user-user similarity
    to generate personalized movie recommendations.
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
        
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            
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
            
            # Reconstruct user means as Polars DataFrame
            self.user_means = pl.DataFrame(mappings['user_means'])
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise e

    def train_model(self):
        """
        Train the recommendation model using Polars for efficient data processing.
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
        
        self.logger.info("Model training completed successfully")

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
        Generate recommendations using pre-calculated normalized ratings.
        """
        try:
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
            
            if not valid_ratings:
                return {'recommendations': {}}
            
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
            
            # Calculate similarities
            similarities = cosine_similarity(temp_user.reshape(1, -1), self.normalized_ratings)[0]
            
            # Get top similar users
            top_n = 50
            top_user_indices = np.argsort(similarities)[-top_n:]
            top_similarities = similarities[top_user_indices]
            
            # Log similarity stats
            self.logger.debug(f"Similarity range: {similarities.min():.2f} to {similarities.max():.2f}")
            self.logger.debug(f"Top 5 similarities: {top_similarities[-5:]}")
            
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
            predicted_ratings = np.zeros_like(weighted_sum)
            predicted_ratings[mask] = weighted_sum[mask] / similarity_sum[mask]
            
            # Scale predictions while preserving relative differences
            nonzero_mask = predicted_ratings > 0
            if nonzero_mask.any():
                # Get prediction statistics
                pred_min = predicted_ratings[nonzero_mask].min()
                pred_max = predicted_ratings[nonzero_mask].max()
                pred_mean = predicted_ratings[nonzero_mask].mean()
                pred_std = predicted_ratings[nonzero_mask].std()
                
                if pred_std > 0:
                    # Use z-score transformation then scale to target range
                    z_scores = (predicted_ratings[nonzero_mask] - pred_mean) / pred_std
                    # Scale to 3.5-5.0 range with sigmoid-like squashing
                    scaled_ratings = 3.5 + 1.5 / (1 + np.exp(-z_scores))
                    predicted_ratings[nonzero_mask] = scaled_ratings
                else:
                    # If no variation, scale linearly
                    predicted_ratings[nonzero_mask] = 3.5 + 1.5 * (
                        (predicted_ratings[nonzero_mask] - pred_min) / 
                        (pred_max - pred_min if pred_max > pred_min else 1)
                    )
            
            # Convert to recommendations
            recommendations = {}
            for idx, rating in enumerate(predicted_ratings):
                if rating > 3.5:
                    movie_id = self.reverse_movie_id_map[idx]
                    if movie_id not in rated_movies:
                        recommendations[str(movie_id)] = round(float(rating), 2)
            
            # Log final recommendations stats
            self.logger.debug(f"Generated {len(recommendations)} recommendations above threshold")
            if recommendations:
                ratings_list = list(recommendations.values())
                self.logger.debug(f"Final ratings range: {min(ratings_list):.2f} to {max(ratings_list):.2f}")
            
            return {
                'recommendations': dict(sorted(
                    recommendations.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:n_recommendations])
            }
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {'recommendations': {}}

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
