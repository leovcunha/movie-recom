const path = require('path');
const webpack = require('webpack');

module.exports = {
  entry: './src/frontend/app.js',

  output: {
    path: __dirname,
    filename: './src/main/resources/static/built/app-bundle.js',
  }, 

  devServer: {

    contentBase: './src/main/resources/',

    proxy: {  
     '/api/*': {
       target: 'http://localhost:8080',
       secure: false
     
      }   
    }
    
   },
   module: {
  rules: [
    {
      test: /\.js$/,
      exclude: /(node_modules|bower_components)/,
      use: {
        loader: 'babel-loader',
        options: {
          presets: ['es2015', 'react']
        }
      }
    }
  ]
}

  
};