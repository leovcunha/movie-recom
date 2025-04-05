import React from 'react';
import { Navbar, Nav, Container } from 'react-bootstrap';
import { Link, useLocation } from 'react-router-dom';
import MovieSearch from './MovieSearch';
import './Header.css';

export default function Header({ showRecommendations, onRate }) {
  const location = useLocation();
  
  return (
    <Navbar expand="lg" className="custom-navbar">
      <Container>
        <Navbar.Brand as={Link} to="/" className="brand">Movie Recommendations</Navbar.Brand>
        
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <div className="search-container d-none d-md-block mx-auto">
            <MovieSearch onRate={onRate} />
          </div>
          
          <Nav className="ms-auto">
            <Nav.Link 
              as={Link} 
              to="/" 
              className={`nav-item ${location.pathname === '/' ? 'active' : ''}`}
            >
              Home
            </Nav.Link>
            <Nav.Link 
              as={Link} 
              to="/recommendations" 
              className={`nav-item ${location.pathname === '/recommendations' ? 'active' : ''}`}
            >
              Recommendations
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
        
        {/* Mobile search - only visible on small screens */}
        <div className="mobile-search-container d-md-none mt-3 w-100">
          <MovieSearch onRate={onRate} />
        </div>
      </Container>
    </Navbar>
  );
}
