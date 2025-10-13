import axios from 'axios';

export const searchMovies = async (query, page = 1) => {
    try {
        const response = await axios.get(`/api/movies/search`, {
            params: { query, page }
        });
        return response;
    } catch (error) {
        console.error('Error searching movies:', error);
        throw error;
    }
}; 