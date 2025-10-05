import axios from 'axios';

export const fetchAvailableMovies = async () => {
    try {
        const response = await axios.get('/api/available-movies');
        return response.data.movie_ids;
    } catch (error) {
        console.error('Error fetching available movies:', error);
        return [];
    }
};