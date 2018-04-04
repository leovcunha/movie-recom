import React from 'react';
import LoginForm from './LoginForm'

export default function Header() {
    return (
      <div>
        <header>
           <h1>Movie Recommendations</h1>
           <LoginForm />
        </header>  
      </div>
    );
    
}

