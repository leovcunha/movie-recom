import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
import logging
import os
import pickle


class MovieRecommender:
    """
    A collaborative filtering-based movie recommendation system that uses user-user similarity
    to generate personalized movie recommendations.
    """

    def __init__(self, ratings_file: str):
        """
        Initialize the MovieRecommender with a ratings dataset.

        Args:
            ratings_file (str): Path to the CSV file containing movie ratings
                              Expected format: userId,movieId,rating
        """
        self.logger = logging.getLogger(__name__)
        self.model_file = "recommender_model.pkl"
        self.ratings_file = ratings_file

        try:
            # Clean slate: remove existing model to ensure fresh training
            if os.path.exists(self.model_file):
                os.remove(self.model_file)
                self.logger.info("Deleted existing model file")

            self.logger.info("Training new model...")
            self.train_model()
            self.log_data_statistics()
            self.logger.info("MovieRecommender initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing MovieRecommender: {str(e)}")
            raise e

    def train_model(self):
        """
        Train the recommendation model using the following steps:
        1. Load and preprocess ratings data
        2. Create movie ID mapping for efficient matrix operations
        3. Build user-movie rating matrix
        4. Calculate user similarity matrix
        """
        # Step 1: Load ratings data
        self.logger.info("Loading ratings data...")
        self.ratings_df = pd.read_csv(
            self.ratings_file,
            names=["userId", "movieId", "rating"],
            dtype={"userId": int, "movieId": int, "rating": float},
        )

        # Handle duplicate ratings by taking the mean
        self.logger.info("Processing ratings and handling duplicates...")
        self.ratings_df = (
            self.ratings_df.groupby(["userId", "movieId"])["rating"]
            .mean()
            .reset_index()
        )

        # Step 2: Create movie ID mapping
        # Map potentially sparse movie IDs to dense sequential indices for efficient matrix operations
        unique_movies = sorted(self.ratings_df["movieId"].unique())
        self.movie_id_map = {
            movie_id: idx for idx, movie_id in enumerate(unique_movies)
        }
        self.reverse_movie_id_map = {
            idx: movie_id for movie_id, idx in self.movie_id_map.items()
        }

        # Apply mapping to movie IDs
        self.ratings_df["movieId_idx"] = self.ratings_df["movieId"].map(
            self.movie_id_map
        )

        # Step 3: Create user-movie matrix
        self.logger.info("Creating user-movie matrix...")
        # Pivot table creates a matrix where rows are users and columns are movies
        self.user_movie_matrix = self.ratings_df.pivot(
            index="userId", columns="movieId_idx", values="rating"
        ).fillna(
            0
        )  # Fill missing ratings with 0

        # Store movie IDs for later use
        self.movie_ids = unique_movies

        # Step 4: Calculate user similarity matrix
        self.logger.info("Calculating user similarities...")
        # Normalize ratings by subtracting mean rating for each user
        # This helps account for different rating scales among users
        user_matrix_normalized = self.user_movie_matrix.subtract(
            self.user_movie_matrix.mean(axis=1), axis=0
        ).fillna(0)
        # Calculate cosine similarity between all users
        self.user_similarity = cosine_similarity(user_matrix_normalized)

    def get_popular_movies(self, n: int = 20) -> Dict[int, int]:
        """
        Get the most viewed movies.

        Args:
            n (int): Number of popular movies to return

        Returns:
            Dict[int, int]: Dictionary of {movie_id: view_count}
        """
        # Count views per movie
        movie_views = self.ratings_df["movieId"].value_counts()

        # Get top N movies
        top_movies = movie_views.head(n)

        # Convert to dictionary with original movie IDs
        popular_movies = {
            int(movie_id): int(count) for movie_id, count in top_movies.items()
        }

        return popular_movies

    def get_recommendations(
        self, user_preferences: Dict[int, float], n_recommendations: int = 10
    ) -> Dict[str, float]:
        """
        Generate movie recommendations based on user preferences.

        Args:
            user_preferences (Dict[int, float]): Dictionary of {movie_id: rating}
            n_recommendations (int): Number of recommendations to generate

        Returns:
            Dict[str, float]: Dictionary of {movie_id: predicted_rating}
        """
        try:
            # Step 1: Validate and map input movie IDs to internal indices
            valid_preferences = {}
            for movie_id, rating in user_preferences.items():
                if movie_id in self.movie_id_map:
                    valid_preferences[self.movie_id_map[movie_id]] = float(rating)

            if not valid_preferences:
                self.logger.error("No valid movie IDs provided")
                return {}

            self.logger.info(f"Valid preferences (mapped): {valid_preferences}")

            # Step 2: Create and normalize temporary user profile
            # Initialize profile with zeros and explicit dtype to avoid warnings
            temp_user_profile = pd.Series(
                0.0, index=range(len(self.movie_id_map)), dtype=float
            )
            for movie_idx, rating in valid_preferences.items():
                temp_user_profile.iloc[movie_idx] = float(rating)

            # Apply z-score normalization to account for rating scale differences
            user_mean = np.mean(list(valid_preferences.values()))
            user_std = np.std(list(valid_preferences.values())) or 1.0
            temp_user_profile = (temp_user_profile - user_mean) / user_std

            # Step 3: Calculate similarity with all users
            # Normalize user matrix using z-scores
            user_matrix_normalized = (
                self.user_movie_matrix.subtract(
                    self.user_movie_matrix.mean(axis=1), axis=0
                )
                .div(self.user_movie_matrix.std(axis=1).fillna(1), axis=0)
                .fillna(0)
            )

            # Calculate cosine similarity between new user and all existing users
            user_sim = cosine_similarity(
                temp_user_profile.values.reshape(1, -1), user_matrix_normalized
            )[0]
            print(user_sim)

            # Step 4: Select similar users
            MIN_SIMILARITY = 0.1  # Minimum similarity threshold
            similar_mask = user_sim >= MIN_SIMILARITY
            top_n_users = min(20, max(sum(similar_mask), 20))  # Take at least 20 users
            similar_users_idx = np.argsort(user_sim)[-top_n_users:]
            similar_users = self.user_movie_matrix.iloc[similar_users_idx]

            # Log similarity scores for debugging
            top_similarities = user_sim[similar_users_idx]
            self.logger.info(f"Top similarity scores: {top_similarities}")

            # Step 5: Calculate predicted ratings
            # Square similarities to give more weight to more similar users
            sim_weights = np.power(user_sim[similar_users_idx], 2).reshape(-1, 1)
            weighted_ratings = similar_users.multiply(sim_weights, axis=0)
            predicted_ratings = weighted_ratings.sum(axis=0) / (
                sim_weights.sum() + 1e-10
            )

            # Remove already rated movies
            predicted_ratings[list(valid_preferences.keys())] = 0

            # Step 6: Scale and filter predictions
            # Scale predictions to 1-5 range
            min_rating, max_rating = 1.0, 5.0
            predicted_ratings = (predicted_ratings - predicted_ratings.min()) / (
                predicted_ratings.max() - predicted_ratings.min() + 1e-10
            )
            predicted_ratings = (
                predicted_ratings * (max_rating - min_rating) + min_rating
            )

            # Generate final recommendations
            MIN_RATING_THRESHOLD = (
                2.0  # Only recommend movies with high predicted ratings
            )
            recommendations = {}
            top_indices = (
                predicted_ratings[predicted_ratings >= MIN_RATING_THRESHOLD]
                .nlargest(n_recommendations)
                .index
            )

            for idx in top_indices:
                original_movie_id = self.reverse_movie_id_map[idx]
                rating = float(predicted_ratings[idx])
                if rating >= MIN_RATING_THRESHOLD:
                    recommendations[str(original_movie_id)] = rating

            self.logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {}

    def log_data_statistics(self):
        """
        Log important statistics about the loaded dataset for monitoring and debugging.
        """
        self.logger.info(f"Total number of ratings: {len(self.ratings_df)}")
        self.logger.info(f"Number of unique movies: {len(self.movie_ids)}")
        self.logger.info(f"Number of unique users: {len(self.user_movie_matrix.index)}")
        self.logger.info(
            f"Movie ID range: {min(self.movie_ids)} to {max(self.movie_ids)}"
        )
        rating_stats = self.ratings_df["rating"].describe()
        self.logger.info(f"Rating statistics:\n{rating_stats}")
