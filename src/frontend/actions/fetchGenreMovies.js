import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const fetchGenreMovies = async (genreId, page = 1) => {
    try {
        const response = await axios.get(`${BACKEND_URL}/api/movies/genre/${genreId}`, {
            params: { page }
        });
        return response;
    } catch (error) {
        console.error('Error fetching genre movies:', error);
        throw error;
    }
}; 