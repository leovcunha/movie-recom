import React, { useState } from 'react';
import { FaStar } from 'react-icons/fa';
import './StarRating.css';

const StarRating = ({ onRate, initialRating = 0, size = 20 }) => {
  const [rating, setRating] = useState(initialRating);
  const [hover, setHover] = useState(null);

  const handleRate = (currentRating) => {
    setRating(currentRating);
    onRate(currentRating);
  };

  return (
    <div className="stars-container">
      {[...Array(5)].map((_, index) => {
        const ratingValue = index + 1;
        return (
          <FaStar
            key={index}
            className="star"
            size={size}
            color={ratingValue <= (hover || rating) ? "#ffc107" : "#e4e5e9"}
            onClick={() => handleRate(ratingValue)}
            onMouseEnter={() => setHover(ratingValue)}
            onMouseLeave={() => setHover(null)}
          />
        );
      })}
    </div>
  );
};

export default StarRating; 