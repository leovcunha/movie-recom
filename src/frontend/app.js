import React from 'react';
import ReactDOM from 'react-dom';
import mdb from './config/mdbkey';

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
        date: new Date(),
        mkey: mdb.key
    }
  }
componentDidMount() {
}
componentWillUnmount() {
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
        <h2>key is {this.state.mkey}  </h2>
      </div>
    );
  }
}

ReactDOM.render(
	<App />,
	document.getElementById('react')
)
