import React, { useState, useEffect } from 'react';
import { Modal, Button, Spinner } from 'react-bootstrap';
import { fetchMovieVideos } from '../actions/fetchMovieVideos';
import StarRating from './StarRating';
import './MovieModal.css';

const MovieModal = ({ show, onHide, movie, onRate }) => {
    const [videos, setVideos] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const loadVideos = async () => {
            if (show && movie) {
                try {
                    setLoading(true);
                    const response = await fetchMovieVideos(movie.id);
                    setVideos(response.data.results);
                } catch (err) {
                    setError('Could not load trailer');
                    console.error(err);
                } finally {
                    setLoading(false);
                }
            }
        };

        loadVideos();
    }, [show, movie]);

    if (!movie) return null;

    const trailer = videos && videos.length > 0 ? videos[0] : null;
    const releaseYear = movie.release_date ? new Date(movie.release_date).getFullYear() : '';

    return (
        <Modal show={show} onHide={onHide} size="lg" centered className="movie-modal">
            <Modal.Header closeButton>
                <Modal.Title>{movie.title} {releaseYear && `(${releaseYear})`}</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <div className="row">
                    <div className="col-md-4">
                        <img 
                            src={`https://image.tmdb.org/t/p/w500${movie.poster_path}`} 
                            alt={movie.title} 
                            className="img-fluid movie-poster"
                        />
                        <div className="rating-container mt-3 d-flex justify-content-center">
                            <StarRating
                                initialRating={movie.userRating || 0}
                                onRate={(rating) => onRate(movie.id, rating)}
                                size={28}
                            />
                        </div>
                    </div>
                    <div className="col-md-8">
                        <h5>Overview</h5>
                        <p>{movie.overview}</p>
                        <div className="trailer-container">
                            {loading ? (
                                <div className="text-center p-5">
                                    <Spinner animation="border" />
                                </div>
                            ) : error ? (
                                <div className="alert alert-warning">{error}</div>
                            ) : trailer ? (
                                <div className="ratio ratio-16x9">
                                    <iframe
                                        src={`https://www.youtube.com/embed/${trailer.key}`}
                                        title={trailer.name}
                                        allowFullScreen
                                    ></iframe>
                                </div>
                            ) : (
                                <div className="alert alert-info">No trailer available</div>
                            )}
                        </div>
                    </div>
                </div>
            </Modal.Body>
            <Modal.Footer>
                <Button variant="secondary" onClick={onHide}>
                    Close
                </Button>
            </Modal.Footer>
        </Modal>
    );
};

export default MovieModal; 