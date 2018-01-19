const packageJSON = require('./package.json');
const path = require('path');
const webpack = require('webpack');

const PATHS = {
  build: path.join(__dirname, 'target', 'classes', 'META-INF', 'resources', 'webjars', packageJSON.name, packageJSON.version)
};

module.exports = {
  entry: './app/index.js',

  output: {
    path: PATHS.build,
    filename: 'app-bundle.js'
  }
};