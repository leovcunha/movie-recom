const greeter = require('greeter');

const greeting = greeter.greet();

if (typeof document !== 'undefined') {
  const el = document.createElement('h1');
  el.innerHTML = greeting;
  document.body.appendChild(el);
} else {
  console.log(greeting);
}