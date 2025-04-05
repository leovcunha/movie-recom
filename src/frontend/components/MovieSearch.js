import React, { useState, useEffect, useRef } from 'react';
import { Form, InputGroup, Button, Spinner } from 'react-bootstrap';
import { searchMovies } from '../actions/searchMovies';
import StarRating from './StarRating';
import MovieModal from './MovieModal';
import { FaSearch, FaTimes } from 'react-icons/fa';
import './MovieSearch.css';

const MovieSearch = ({ onRate }) => {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [showOverlay, setShowOverlay] = useState(false);
    const [selectedMovie, setSelectedMovie] = useState(null);
    const [showModal, setShowModal] = useState(false);
    const searchRef = useRef(null);
    const inputRef = useRef(null);
    const debounceTimeout = useRef(null);

    // Close overlay when pressing Escape
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                setShowOverlay(false);
            }
        };

        window.addEventListener('keydown', handleEscape);
        return () => {
            window.removeEventListener('keydown', handleEscape);
        };
    }, []);

    // Focus input when overlay opens
    useEffect(() => {
        if (showOverlay && inputRef.current) {
            inputRef.current.focus();
        }
    }, [showOverlay]);

    const handleSearch = (e) => {
        e.preventDefault();
        if (query.length >= 2) {
            setShowOverlay(true);
            performSearch();
        }
    };

    const performSearch = async () => {
        if (query.length < 2) return;
        
        try {
            setIsLoading(true);
            const response = await searchMovies(query);
            setResults(response.data.results);
        } catch (error) {
            console.error('Search error:', error);
        } finally {
            setIsLoading(false);
        }
    };

    const clearSearch = () => {
        setQuery('');
        setResults([]);
    };

    const closeOverlay = () => {
        setShowOverlay(false);
    };

    const handleInputChange = (e) => {
        const newQuery = e.target.value;
        setQuery(newQuery);
        
        if (showOverlay && debounceTimeout.current) {
            clearTimeout(debounceTimeout.current);
        }

        if (showOverlay && newQuery.length >= 2) {
            debounceTimeout.current = setTimeout(() => {
                performSearch();
            }, 500);
        }
    };

    const handleRate = (movieId, rating, event) => {
        if (event) {
            event.stopPropagation(); // Prevent opening modal when rating
        }
        onRate(movieId, rating);
    };

    const handleMovieClick = (movie) => {
        setSelectedMovie(movie);
        setShowModal(true);
    };

    const handleCloseModal = () => {
        setShowModal(false);
    };

    return (
        <>
            <div className="movie-search" ref={searchRef}>
                <Form onSubmit={handleSearch}>
                    <InputGroup>
                        <Form.Control
                            ref={inputRef}
                            type="text"
                            placeholder="Search for a movie..."
                            value={query}
                            onChange={handleInputChange}
                            aria-label="Search"
                            className="search-input"
                        />
                        {query && (
                            <Button 
                                variant="outline-secondary" 
                                onClick={clearSearch}
                                className="clear-button"
                            >
                                <FaTimes />
                            </Button>
                        )}
                        <Button 
                            variant="primary" 
                            type="submit"
                            disabled={query.length < 2 || isLoading}
                            className="search-button"
                        >
                            <FaSearch />
                        </Button>
                    </InputGroup>
                </Form>
            </div>

            {/* Full screen overlay for search results */}
            {showOverlay && (
                <div className="search-overlay">
                    <div className="search-overlay-header">
                        <div className="search-overlay-title">
                            Search Results: {query}
                        </div>
                        <button className="close-search-btn" onClick={closeOverlay}>
                            <FaTimes />
                        </button>
                    </div>
                    
                    <div className="search-overlay-content">
                        <Form onSubmit={handleSearch}>
                            <InputGroup className="mb-4">
                                <Form.Control
                                    type="text"
                                    placeholder="Search for a movie..."
                                    value={query}
                                    onChange={handleInputChange}
                                    aria-label="Search"
                                    className="search-input"
                                    autoFocus
                                />
                                {query && (
                                    <Button 
                                        variant="outline-secondary" 
                                        onClick={clearSearch}
                                        className="clear-button"
                                    >
                                        <FaTimes />
                                    </Button>
                                )}
                                <Button 
                                    variant="primary" 
                                    type="submit"
                                    disabled={query.length < 2 || isLoading}
                                    className="search-button"
                                >
                                    <FaSearch />
                                </Button>
                            </InputGroup>
                        </Form>

                        {isLoading ? (
                            <div className="search-loading">
                                <Spinner animation="border" variant="light" />
                                <p className="mt-3">Searching for movies...</p>
                            </div>
                        ) : results.length > 0 ? (
                            <div className="search-results-grid">
                                {results.map(movie => (
                                    <div 
                                        key={movie.id} 
                                        className="movie-card search-movie-card" 
                                        onClick={() => handleMovieClick(movie)}
                                    >
                                        <div className="position-relative">
                                            <img 
                                                src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`}
                                                alt={movie.title}
                                                className="card-img-top"
                                            />
                                            <div className="rating-container">
                                                <StarRating
                                                    initialRating={movie.userRating || 0}
                                                    onRate={(rating) => handleRate(movie.id, rating, event)}
                                                    size={24}
                                                />
                                            </div>
                                        </div>
                                        <div className="card-body">
                                            <h5 className="card-title">{movie.title}</h5>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : query.length >= 2 ? (
                            <div className="search-no-results">
                                No movies found for "{query}"
                            </div>
                        ) : null}
                    </div>
                </div>
            )}

            {/* Movie details modal */}
            <MovieModal 
                show={showModal} 
                onHide={handleCloseModal} 
                movie={selectedMovie}
                onRate={handleRate}
            />
        </>
    );
};

export default MovieSearch; 