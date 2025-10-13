# Movie Recommendation Project

This is a movie recommendation project built using FastAPI for the backend and React for the frontend.

## Backend

The backend is powered by FastAPI and uses collaborative filtering approach for recommendations. The main file is located at `src/main/python/app.py`.

## Frontend

The frontend is built with React and can be found in the `src/frontend` directory.

## Getting Started

1. **Backend**

    - Create a conda environment: `conda create -n movie-rec python=3.10`
    - Activate the conda environment: `conda activate movie-rec`
    - Install dependencies: `pip install -r requirements.txt`
    - cd src/backend
    - run `uvicorn app:app --reload`
    - API documentation available at: `http://localhost:8000/docs`
    - download data from [here](https://drive.google.com/file/d/1KNHvgPM8HupZl6XNrd5in0Cw6WL3-5FW/view?usp=drive_link) into `src/backend/data` 

2. **Frontend**
    - Navigate to frontend directory: `cd src/frontend`
    - Install dependencies: `npm install`
    - Build the project: `npm run build`
    - Start the development server: `npm start`


## Key Components of the Recommendation Engine

1. User-Based Collaborative Filtering  
This approach finds users similar to the active user and recommends movies that those similar users liked. Ratings are normalized to account for different rating scales between users. Cosine similarity is used to find similar users based on their rating pattern. Weighted averaging of ratings from similar users to predict ratings for unseen movies.  

2. Item-Based Collaborative Filtering  
This approach recommends movies similar to those the user has already rated highly. The system pre-calculates similarities between the most popular movies to speed up recommendations.  Movies are considered similar if they are rated similarly by the same users. The systm uses caching and sparse matrices for efficient processing. 

3. Hybrid Approach  
The final recommendations combine both user-based and item-based predictions: 40% user-based + 60% item-based for a balanced approach.
Final predictions are scaled to a 3.5-5.0 range using statistical transformations. Only recommends movies with predicted ratings above 3.5.

4. API Recommendations Cold Start    
For movies not in the dataset or when not enough recommendations are generated:  
    * TMDB API Integration: Fetches similar proxy movies from The Movie Database API until a movie in our recommendation database is found
    * Caching: API results are cached to reduce API calls
    * The proxy movies are used to generate recommendations

## Memory Optimization

I had to spend tons of time to try to optimize memory as I wanted to deploy this in the lowest RAM memory machine as possible. The model is large considering the number of user_ratings used for training is >1M rows. I ended up having to use different approaches:

1. Proactive Memory Checking and Dynamic Optimization Modes

- The system doesn't just hope for the best; it actively checks the available memory and adjusts its behavior accordingly.

- On Startup (app.py): Before even loading the recommender, app.py checks the available system memory using `psutil`. If it's below a certain threshold (100MB), it sets an environment variable EXTREME_MEMORY_OPTIMIZATION to "1".
- During Model Loading (recommendation_service.py): The load_model method reads this environment variable and also re-checks available memory. This sets two internal flags:
low_memory_mode: Activated if available memory is low.
extreme_optimization: A more aggressive mode triggered by the environment variable.
These flags dynamically change how the model is loaded and used, making it highly adaptable to its environment.

2. Strategic and Partial Data Loading
Instead of loading all model files at once, load_model does it in a careful, memory-aware sequence, from smallest to largest, with checks and garbage collection along the way.

- Mappings First: It loads the mappings.pkl file first, which contains the user and movie ID dictionaries. These are essential and relatively small.
- Selective Caching: The top_similar_movies cache, which can be large, is loaded very selectively. In low_memory_mode or extreme_optimization, it only keeps similarities for the top 200-500 most popular movies, and for each of those, it only keeps the top 3-5 most similar items. The full cache is loaded into a temporary variable and then immediately discarded.
Skipping Large Matrices: In low_memory_mode, the service completely skips loading the large sparse matrices (normalized_ratings, user_movie_matrix, etc.) and instead creates empty placeholder matrices. This is a crucial fallback that prevents the application from crashing due to memory allocation failure, allowing it to run in a degraded but functional state (likely relying more on item-based logic).

3. Use of Memory-Efficient Data Structures
The choice of data structures is critical for managing memory.

- Sparse Matrices (scipy.sparse.csr_matrix): The user-item rating matrix is naturally sparse (most users have not rated most movies). Instead of a dense NumPy array that would store millions of zeros, a csr_matrix only stores the non-zero ratings and their locations. This reduces memory usage by orders of magnitude.
Efficient Data Types (np.float32): In get_recommendations, arrays for predictions are created using np.float32 instead of the default np.float64. This cuts the memory required for these arrays in half with a negligible impact on prediction accuracy.
4. Explicit Garbage Collection and Object Deletion
The code doesn't rely solely on Python's automatic garbage collector.

- `gc.collect()`:  This is an explicit instruction to Python to reclaim memory from objects that are no longer referenced. It's used strategically after loading a large file or deleting a large variable to free up memory immediately.
`del` Keyword: Large intermediate variables, like the similarities matrix in load_model or the combined_predictions array in get_recommendations, are explicitly deleted with del as soon as they are no longer needed. This makes them immediately available for garbage collection.


5. Batch Processing for Computations
- Calculating the similarity of a new user against hundreds of thousands of existing users can create a massive intermediate result.

- Batched Cosine Similarity: In get_recommendations, instead of calculating the cosine similarity against the entire normalized_ratings matrix at once, it's done in batches of 1000 users. This keeps the memory footprint of the calculation small and predictable, preventing a single large allocation that could crash the application.


## Contributing

Contributions are welcome! Please open an issue to discuss any changes or improvements.
