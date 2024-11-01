import axios from 'axios';

const TMDB_API_KEY = 'api_key=9624561704e52e84ae59cd0147eb662d';
const BACKEND_URL = 'http://localhost:8000'; // Adjust if your backend URL is different

export const fetchPopularMovies = async () => {
    try {
        // Step 1: Get popular movie IDs from your backend
        const backendResponse = await axios.get(`${BACKEND_URL}/valid-movies`);
        const popularMovies = backendResponse.data.popular_movies;

        // Step 2: Get movie details from TMDB
        const moviePromises = Object.keys(popularMovies).map(async (movieId) => {
            const tmdbResponse = await axios.get(
                `https://api.themoviedb.org/3/movie/${movieId}?${TMDB_API_KEY}&language=en-US`
            );
            return tmdbResponse.data;
        });

        const movieDetails = await Promise.all(moviePromises);
        return {
            data: {
                results: movieDetails,
            },
        };
    } catch (error) {
        console.error('Error fetching popular movies:', error);
        throw error;
    }
};
