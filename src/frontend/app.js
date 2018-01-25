const React = require('react');
const ReactDOM = require('react-dom');


class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {date: new Date()};
  }
componentDidMount() {
}
componentWillUnmount() {
}

render() {
    return (
      <div>
        <h1>Hello, world!</h1>
        <h2>It is {this.state.date.toLocaleTimeString()}.</h2>
      </div>
    );
  }
}

ReactDOM.render(
	<App />,
	document.getElementById('react')
)
