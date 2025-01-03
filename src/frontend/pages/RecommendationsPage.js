import React, { useEffect, useState } from 'react';
import { Container, Alert, Row, Col, Card } from 'react-bootstrap';
import StarRating from '../components/StarRating';
import axios from 'axios';
import './RecommendationsPage.css';

const RecommendationsPage = ({ onRate }) => {
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRecommendations();
  }, []);

  const fetchRecommendations = async () => {
    console.log('Starting recommendations fetch...');
    try {
      // Clear existing recommendations and show loading state
      setRecommendations([]);
      setLoading(true);
      setError(null);
      console.log('Cleared recommendations and set loading state');

      const userRatings = JSON.parse(localStorage.getItem('userRatings') || '{}');
      console.log('Loaded user ratings:', userRatings);
      
      // Check if there are at least 5 ratings
      if (Object.keys(userRatings).length < 5) {
        console.log('Less than 5 ratings, skipping fetch');
        setLoading(false);
        return;
      }

      // Convert ratings to numbers
      const ratings = {};
      Object.entries(userRatings).forEach(([key, value]) => {
        ratings[parseInt(key)] = parseFloat(value);
      });
      console.log('Converted ratings:', ratings);

      console.log('Making API request...');
      const response = await axios.post('/api/recommendations', {
        ratings: ratings
      });
      console.log('Received response:', response.data);

      // Filter out any movies that have been rated
      const filteredRecommendations = response.data.filter(
        movie => !ratings.hasOwnProperty(movie.id)
      );
      
      setRecommendations(filteredRecommendations);
      console.log('Updated recommendations after filtering:', filteredRecommendations);
    } catch (err) {
      console.error('Error fetching recommendations:', err);
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRate = (movieId, rating) => {
    onRate(movieId, rating);
    
    // Update the movie's rating in the recommendations list
    setRecommendations(prevRecommendations => 
      prevRecommendations.map(movie => 
        movie.id === movieId ? { ...movie, userRating: rating } : movie
      )
    );
  };

  const handleReset = () => {
    localStorage.removeItem('userRatings');
    setRecommendations([]);
    // Optionally refresh the page to start completely fresh
    window.location.reload();
  };

  if (loading) {
    return (
      <Container className="recommendations-container">
        <div className="loading">Loading recommendations...</div>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="recommendations-container">
        <Alert variant="danger">
          {error}
        </Alert>
      </Container>
    );
  }

  return (
    <Container className="recommendations-container">
      <div className="recommendations-header">
        <h1>Your Personalized Recommendations</h1>
        <p>Based on your movie ratings</p>
        {recommendations.length > 0 && (
          <>
            <button 
              className="btn btn-primary mb-3 me-2" 
              onClick={fetchRecommendations}
            >
              Refresh Recommendations
            </button>
            <button 
              className="btn btn-warning mb-3" 
              onClick={handleReset}
            >
              Reset All Ratings
            </button>
          </>
        )}
      </div>
      
      {recommendations.length > 0 ? (
        <Row className="g-4">
          {recommendations.map((movie) => (
            <Col key={movie.id} xs={12} sm={6} md={3}>
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
                  <div className="predicted-rating">
                    Predicted Rating: {movie.predicted_rating.toFixed(1)}
                  </div>
                </Card.Body>
              </Card>
            </Col>
          ))}
        </Row>
      ) : (
        <Alert variant="info">
          Rate at least 5 movies to get personalized recommendations!
        </Alert>
      )}
    </Container>
  );
};

export default RecommendationsPage;
