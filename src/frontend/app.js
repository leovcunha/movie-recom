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
        isLoading: false,
        movieList: {
            results: []
        }

    };
  }
componentDidMount() {
    this.setState({isLoading: true})
    discoverMovies().then(res => {
        const movieList = res.data;
        this.setState({ movieList, isLoading: false });
    });   
}

render() {

    console.log(this.state.movieList)
    
    return (
      <div>
        <Header />
        {this.state.isLoading? ( //if (isLoading)
             <p>Loading...</p>
             ) : (    //else
             <MovieTable movies={this.state.movieList.results} />
             )
         }
      </div>
    );
  }
}

ReactDOM.render(
	<App />,
	document.getElementById('react')
)
