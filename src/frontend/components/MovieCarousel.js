import React, { useState, useEffect } from 'react';
import { Carousel, Row, Col, Card } from 'react-bootstrap';
import StarRating from './StarRating';
import './MovieCarousel.css';

const MovieCarousel = ({ movies, title, onLoadMore, currentPage, totalPages, onRate }) => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [isLoading, setIsLoading] = useState(false);
    const [hasLoadedMore, setHasLoadedMore] = useState(false);

    // Calculate number of movies per slide based on screen size
    const getMoviesPerSlide = () => {
        if (window.innerWidth < 576) return 2; // Mobile
        if (window.innerWidth < 768) return 3; // Tablet
        return 5; // Desktop
    };

    const moviesPerSlide = getMoviesPerSlide();
    const movieGroups = [];
    
    // Group movies into sets for each carousel slide
    for (let i = 0; i < movies.length; i += moviesPerSlide) {
        movieGroups.push(movies.slice(i, i + moviesPerSlide));
    }

    const handleSelect = async (selectedIndex) => {
        setCurrentSlide(selectedIndex);
        
        // Load more only when we reach the last slide
        if (selectedIndex === movieGroups.length - 1 && 
            onLoadMore && 
            currentPage < totalPages && 
            !isLoading && 
            !hasLoadedMore) {
            setIsLoading(true);
            setHasLoadedMore(true);
            await onLoadMore();
            setIsLoading(false);
            // Reset hasLoadedMore after a delay
            setTimeout(() => setHasLoadedMore(false), 1500);
        }
    };

    // Reset hasLoadedMore when page changes
    useEffect(() => {
        setHasLoadedMore(false);
    }, [currentPage]);

    const handleRate = (movieId, rating) => {
        onRate(movieId, rating);
    };

    return (
        <div className="movie-section mb-5">
            <h2 className="mb-3">{title}</h2>
            <Carousel 
                interval={null} 
                onSelect={handleSelect}
                activeIndex={currentSlide}
                wrap={false}
            >
                {movieGroups.map((group, index) => (
                    <Carousel.Item key={index}>
                        <Row>
                            {group.map((movie) => (
                                <Col key={movie.id} className="movie-col">
                                    <Card className="movie-card h-100">
                                        <div className="position-relative">
                                            <Card.Img
                                                variant="top"
                                                src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`}
                                                alt={movie.title}
                                            />
                                            <div className="rating-container">
                                                <StarRating
                                                    initialRating={movie.userRating || 0}
                                                    onRate={(rating) => handleRate(movie.id, rating)}
                                                    size={24}
                                                />
                                            </div>
                                        </div>
                                        <Card.Body>
                                            <Card.Title>{movie.title}</Card.Title>
                                        </Card.Body>
                                    </Card>
                                </Col>
                            ))}
                        </Row>
                    </Carousel.Item>
                ))}
            </Carousel>
        </div>
    );
};

export default MovieCarousel;
