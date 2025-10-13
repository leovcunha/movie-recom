import axios from 'axios';

export const fetchPopularMovies = async (page = 1) => {
    try {
        const response = await axios.get(`/api/movies/popular`, {
            params: { page }
        });
        return response;
    } catch (error) {
        console.error('Error fetching popular movies:', error);
        throw error;
    }
};
