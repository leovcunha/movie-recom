/*eslint-env es_modules */
import axios from 'axios';

export const discoverMovies = function (page) {
    return axios.get(`/api/movies/discover`, {
        params: { page },
    });
};
