import React, { useState } from 'react';
import { Carousel, Row, Col, Card } from 'react-bootstrap';

const MovieCarousel = ({ movies, title, onLoadMore, currentPage, totalPages }) => {
    const [currentSlide, setCurrentSlide] = useState(0);
    const [isLoading, setIsLoading] = useState(false);

    // Group movies into sets of 4 for each carousel slide
    const movieGroups = [];
    for (let i = 0; i < movies.length; i += 4) {
        movieGroups.push(movies.slice(i, i + 4));
    }

    const handleSelect = async (selectedIndex) => {
        setCurrentSlide(selectedIndex);
        
        // If we're at the last slide and there are more pages to load
        if (selectedIndex === movieGroups.length - 1 && onLoadMore && currentPage < totalPages && !isLoading) {
            setIsLoading(true);
            await onLoadMore();
            setIsLoading(false);
        }
    };

    return (
        <div className="movie-section mb-5">
            <h2 className="mb-3">{title}</h2>
            <Carousel 
                interval={null} 
                onSelect={handleSelect}
                activeIndex={currentSlide}
                wrap={false}  // Prevent wrapping to first slide
            >
                {movieGroups.map((group, index) => (
                    <Carousel.Item key={index}>
                        <Row>
                            {group.map((movie) => (
                                <Col key={movie.id} xs={12} sm={6} md={3}>
                                    <Card className="movie-card h-100">
                                        <Card.Img
                                            variant="top"
                                            src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`}
                                            alt={movie.title}
                                        />
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