import React from 'react';
import ReactDOM from 'react-dom';
import { discoverMovies } from './actions/discoverMovies'
import axios from 'axios'
const APIKEY = 'api_key=9624561704e52e84ae59cd0147eb662d'

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
        date: new Date(),
        movieList: {}

    }
  }
componentDidMount() {
    console.log(this.state.movieList);    
}
componentWillMount() {
    axios.get(`https://api.themoviedb.org/3/discover/movie?${APIKEY}&language=en-US&page=1`) .then(res => {
        const movieList = res.data;
        this.setState({ movieList });
    });
}

render() {
    return (
      <div>
        <header>
            <h1>Movie Recommendations</h1>
            <ul id="nav">
                <li className="item1">1</li>
                <li className="item2">2</li>
                <li className="item3">3</li>
            </ul>
        </header>
        <h2>It is {this.state.date.toLocaleTimeString()}.</h2>
        
      </div>
    );
  }
}

ReactDOM.render(
	<App />,
	document.getElementById('react')
)
