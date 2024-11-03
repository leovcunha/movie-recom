/*eslint-env es_modules */
import axios from 'axios';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

export const discoverMovies = function (page) {
    return axios.get(`${BACKEND_URL}/api/movies/discover`, {
        params: { page },
    });
};
