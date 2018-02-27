import React from 'react';
import ReactDOM from 'react-dom';
import { discoverMovies } from './actions/discoverMovies';

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
componentWillMount() {

}

render() {
    console.log(this.state.movieList);
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
