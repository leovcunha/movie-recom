/*eslint-env es_modules */
import axios from 'axios';

const APIKEY = 'api_key=9624561704e52e84ae59cd0147eb662d';

export const discoverMovies = function (pg = 1) {
    return axios.get(`https://api.themoviedb.org/3/discover/movie?${APIKEY}&language=en-US&page={pg}&sort_by=popularity.desc`);   
};