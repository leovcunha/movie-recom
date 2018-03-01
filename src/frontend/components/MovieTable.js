import React from 'react';
import MovieHolder from './MovieHolder';

export default class MovieTable extends React.Component {
    constructor(props) {
        super(props);
    }
    
    listMovies() {
        return (
          this.props.movies.map((movieData) => {
            return <MovieHolder key={movieData.id} movie={movieData}/>             
          })
        )
    }
    
    render() {
        
        return (
            <div className='movietable'>
                {this.listMovies()}
            </div>        
        );
    }
    
    
    
}