import axios from 'axios';

export const AUTH_USER = 'auth_user';
export const UNAUTH_USER = 'unauth_user';
export const AUTH_ERROR = 'auth_error';
export const FETCH_MESSAGE = 'fetch_message';

export function login(username, password) {

  return function(dispatch) {

    axios.post(`/login`, {
        "username": username, 
        "password": password })
      .then(response => {
        dispatch({ type: AUTH_USER });
        localStorage.setItem('token', response.headers.authorization);
      })
      .catch(() => {
        dispatch({ type: AUTH_ERROR, payload: 'bad login info' });
      });

}

export function signup({ email, password }) {
  return function(dispatch) {
    axios.post(`/api/spectators`, { email, password })
      .then(response => {
        dispatch({ type: AUTH_USER });
        localStorage.setItem('token', response.data.token);

      })
      .catch(response => dispatch(
         {
            type: AUTH_ERROR,
            payload: response.data.error
         }));
  }
}

export function logout() {
  localStorage.removeItem('token');

  return { type: UNAUTH_USER };
}

