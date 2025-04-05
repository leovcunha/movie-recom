import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const fetchMovieVideos = async (movieId) => {
    try {
        const response = await axios.get(`${BACKEND_URL}/api/movies/${movieId}/videos`);
        return response;
    } catch (error) {
        console.error('Error fetching movie videos:', error);
        throw error;
    }
}; 