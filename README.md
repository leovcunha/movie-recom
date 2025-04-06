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

## Deployment to Render

### One-Click Deployment

1. Create a Render account at [render.com](https://render.com)
2. Fork this repository to your GitHub account
3. Set the correct repository URL in the `render.yaml` file

### Manual Deployment

1. Push the repository to GitHub
2. Make sure to set execute permissions for the build script:
   ```
   git update-index --chmod=+x render-build.sh
   git commit -m "Make render-build.sh executable"
   ```
3. Connect your GitHub account to Render
4. Create a new Web Service and select your repository
5. Select "Docker" as the environment
6. Add the environment variable `TMDB_API_KEY` with your API key
7. Use the free plan (or choose a paid plan for better performance)
8. Click "Create Web Service"

The deployment will automatically:
- Download the required dataset during build
- Set up the environment variables
- Build and run the application

## Key Components of the Recommendation Engine

1. User-Based Collaborative Filtering  
This approach finds users similar to the active user and recommends movies that those similar users liked. Ratings are normalized to account for different rating scales between users. Cosine similarity is used to find similar users based on their rating pattern. Weighted averaging of ratings from similar users to predict ratings for unseen movies.  

2. Item-Based Collaborative Filtering  
This approach recommends movies similar to those the user has already rated highly. The system pre-calculates similarities between the most popular movies to speed up recommendations.  Movies are considered similar if they are rated similarly by the same users. The systm uses caching and sparse matrices for efficient processing. 

3. Hybrid Approach  
The final recommendations combine both user-based and item-based predictions: 40% user-based + 60% item-based for a balanced approach.
Final predictions are scaled to a 3.5-5.0 range using statistical transformations. Only recommends movies with predicted ratings above 3.5.

4. API Recommendations Fallback    
For movies not in the dataset or when not enough recommendations are generated:  
    * TMDB API Integration: Fetches similar movies from The Movie Database API
    * Caching: API results are cached to reduce API calls
    * Rating Adjustment: External ratings are scaled to match the system's rating scale
5. Model Management
The system efficiently manages its model:  
    * Caching: The trained model is saved to disk to avoid retraining on every startup  
    * Incremental Updates: Only retrains when the underlying data has changed

## Contributing

Contributions are welcome! Please open an issue to discuss any changes or improvements.
