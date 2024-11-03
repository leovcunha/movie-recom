import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const fetchPopularMovies = async () => {
    try {
        const backendResponse = await axios.get(`${BACKEND_URL}/valid-movies`);
        const popularMovies = backendResponse.data.popular_movies;

        const moviePromises = Object.keys(popularMovies).map(async (movieId) => {
            const response = await axios.get(`${BACKEND_URL}/api/movies/${movieId}`);
            return response.data;
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
