import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const fetchMovieDetails = async (movieId) => {
    try {
        const response = await axios.get(`${BACKEND_URL}/api/movie/${movieId}`);
        return response;
    } catch (error) {
        console.error('Error fetching movie details:', error);
        throw error;
    }
};  