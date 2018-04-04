export default function(state = null, action) {

    switch(action.type) {
        case 'USER_LOGGEDIN':
            return action.payload;
    }

    return state;
}