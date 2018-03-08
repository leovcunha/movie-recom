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
            <div className="movietable">
                <i className="material-icons arrow-left">chevron_left</i>       
                <div className="movielist">
                    {this.listMovies()}
                </div>
                <i className="material-icons arrow-right">chevron_right</i>
            </div>
        );
    }
    
    
    
}