import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const searchMovies = async (query, page = 1) => {
    try {
        const response = await axios.get(`${BACKEND_URL}/api/movies/search`, {
            params: { query, page }
        });
        return response;
    } catch (error) {
        console.error('Error searching movies:', error);
        throw error;
    }
}; 