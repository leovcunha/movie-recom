/*eslint-env es_modules */
import React from 'react';
import ReactDOM from 'react-dom';
import { discoverMovies } from './actions/discoverMovies';
import Header from './components/Header';
import MovieTable from './components/MovieTable';

class App extends React.Component {
  constructor(props) {
    super(props);
    
    this.pageHandler = this.pageHandler.bind(this);
    
    this.state = {
        isLoading: false,
        moviePage: 1,
        movieList: {
            results: []
        }

    };
  }
componentDidMount() {
    this.setState({isLoading: true});
    discoverMovies().then(res => {
        const movieList = res.data;
        this.setState({ movieList, isLoading: false });
    });   
}

pageHandler(increm) {
    let mp = this.state.moviePage;
    if increm {
        mp =  mp + 1;
    } 
    else if (mp > 1) {
        mp = mp - 1;
    }

    this.setState({
        moviePage: mp
    }); 
    
}

render() {

    console.log(this.state.movieList);
    
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
