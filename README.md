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
    - Run the FastAPI server: `uvicorn src.main.python.app:app --reload`
    - API documentation available at: `http://localhost:8000/docs`

2. **Frontend**
    - Navigate to frontend directory: `cd src/frontend`
    - Install dependencies: `npm install`
    - Build the project: `npm run build`
    - Start the development server: `npm start`

## Contributing

Contributions are welcome! Please open an issue to discuss any changes or improvements.
