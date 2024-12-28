import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const fetchPopularMovies = async (page = 1) => {
    try {
        const response = await axios.get(`${BACKEND_URL}/api/movies/popular`, {
            params: { page }
        });
        return response;
    } catch (error) {
        console.error('Error fetching popular movies:', error);
        throw error;
    }
};
