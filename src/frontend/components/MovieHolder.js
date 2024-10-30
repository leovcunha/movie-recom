import React from 'react';

export default function MovieHolder(props) {
    return (
        <img className="mov-img" src={`http://image.tmdb.org/t/p/w185${props.movie.poster_path}`} />
    );
}
