import axios from 'axios';

export const fetchMovieDetails = async (movieId) => {
    try {
        const response = await axios.get(`/api/movie/${movieId}`);
        return response;
    } catch (error) {
        console.error('Error fetching movie details:', error);
        throw error;
    }
}; 