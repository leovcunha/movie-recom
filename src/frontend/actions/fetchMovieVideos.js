import axios from 'axios';


export const fetchMovieVideos = async (movieId) => {
    try {
        const response = await axios.get(`/api/movies/${movieId}/videos`);
        return response;
    } catch (error) {
        console.error('Error fetching movie videos:', error);
        throw error;
    }
}; 