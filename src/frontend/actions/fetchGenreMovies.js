import axios from 'axios';

export const fetchGenreMovies = async (genreId, page = 1) => {
    try {
        const response = await axios.get(`/api/movies/genre/${genreId}`, {
            params: { page }
        });
        return response;
    } catch (error) {
        console.error('Error fetching genre movies:', error);
        throw error;
    }
}; 