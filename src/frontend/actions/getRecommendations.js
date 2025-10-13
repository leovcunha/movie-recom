import axios from 'axios';

export const getRecommendations = async (ratings) => {
    try {
        const response = await axios.post(`/api/recommendations`, {
            ratings: ratings
        });
        return response;
    } catch (error) {
        console.error('Error fetching recommendations:', error);
        throw error;
    }
}; 