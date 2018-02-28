/*eslint-env es_modules */
import React from 'react';
import ReactDOM from 'react-dom';
import { discoverMovies } from './actions/discoverMovies';
import Header from './components/Header';
import MovieTable from './components/MovieTable';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
        date: new Date(),
        movieList: {}

    };
  }
componentDidMount() {
    discoverMovies().then(res => {
        const movieList = res.data;
        this.setState({ movieList });
    });   
}

renderMovieTable() {
    let MovieTab = <h4>Loading..</h4>
    if (this.state.movieList != {}) {
        MovieTab = <MovieTable movies={this.state.movieList.results} />;
    }
    return MovieTab;
}

render() {
    console.log(this.state.movieList)

    return (
      <div>
        <Header />
        {this.renderMovieTable()}
      </div>
    );
  }
}

ReactDOM.render(
	<App />,
	document.getElementById('react')
)
