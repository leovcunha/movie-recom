import React from 'react';
import { Carousel, Row, Col, Card } from 'react-bootstrap';

const MovieCarousel = ({ movies, title }) => {
    // Group movies into sets of 4 for each carousel slide
    const movieGroups = [];
    for (let i = 0; i < movies.length; i += 4) {
        movieGroups.push(movies.slice(i, i + 4));
    }

    return (
        <div className="movie-section mb-5">
            <h2 className="mb-3">{title}</h2>
            <Carousel interval={null}>
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
                                            <Card.Text>
                                                Rating: {movie.vote_average}/10
                                            </Card.Text>
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