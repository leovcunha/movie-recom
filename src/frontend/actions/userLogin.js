export function userLogin(userName) {

    return {
        type: 'USER_LOGGEDIN',
        payload: {
            username: userName,
            userLoggedIn: true          
        }
    };

}