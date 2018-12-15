/*eslint-env es_modules */
import React from 'react';
import ReactDOM from 'react-dom';
import { discoverMovies } from './actions/discoverMovies';
import Header from './components/Header';
import MovieTable from './components/MovieTable';

import { Provider } from 'react-redux';
import { createStore, applyMiddleware } from 'redux';
import reducers from './reducers';
import reduxThunk from 'redux-thunk';

const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose;
const createStoreWithMiddleware = composeEnhancers(applyMiddleware(reduxThunk))(createStore);
const store = createStoreWithMiddleware(reducers);

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
    this.triggerMoviesUpdate();
}

triggerMoviesUpdate() {
    this.setState({isLoading: true});
    discoverMovies(this.state.moviePage.toString()).then(res => {
        const movieList = res.data;
        return this.setState({ movieList, isLoading: false });
    });   
}

pageHandler(increm) {
    let mp = this.state.moviePage;
    if (increm) {
        mp =  mp + 1;
    } 
    else if (!increm && mp > 1) {
        mp = mp - 1;
    }

    this.setState({
        moviePage: mp
    }, () => this.triggerMoviesUpdate()); 
       
}

render() {

    return (
      <div>
        <Header />
        {this.state.isLoading? ( //if (isLoading)
             <p>Loading...</p>
             ) : (    //else
             <MovieTable movies={this.state.movieList.results} pagehandler={this.pageHandler}/>
             )
         }
      </div>
    );
  }
}

ReactDOM.render(
    <Provider store={store}>
	   <App />
	</Provider> ,
	document.getElementById('react')
);
