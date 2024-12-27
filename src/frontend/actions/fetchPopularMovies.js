import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const fetchPopularMovies = async () => {
    try {
        const backendResponse = await axios.get(`${BACKEND_URL}/valid-movies`);
        const popularMovies = backendResponse.data.popular_movies;
        const movieDetails = backendResponse.data.movie_details;

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
