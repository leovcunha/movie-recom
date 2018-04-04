import React from 'react';
import { connect } from 'react-redux';

export default class LoginForm extends React.Component {

   constructor(props) {
       super(props);
   }
   
   render() {
    if(props.isLoggedIn) {
        return (
        <p className="login">Hello {this.props.username}!</p>
        <form action="/logout" method="POST">
	         <input type="submit" value="Logout"/>
	    </form>
        )
        
    }
    return (
        <form className="login" action="/login"  method="POST">
          Username:
          <input type="text" name="username" value="">
          Password:
          <input type="text" name="password" value="">
          <input type="submit" value="Submit">
         </form>
    );
   }
    
}

function mapStateToProps(state) {
    return {
      username: state.username, 
      isLoggedIn: state.isLoggedIn
    };
}

export default connect(mapStateToProps)(LoginForm);