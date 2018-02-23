const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: './src/frontend/app.js',

  output: {
    path: path.resolve(__dirname, 'src/main/resources/static/built'),
    filename: 'app-bundle.js',
    publicPath: '/'
  }, 

  plugins: [

    new HtmlWebpackPlugin({ //dev
        hash: true,
        template: 'src/main/resources/templates/template.html',
    }),
    new HtmlWebpackPlugin({ //production
        hash: false,
        template: 'src/main/resources/templates/template.html',
        filename: path.join(__dirname, 'src/main/resources/templates/index.html')
    })
  ],

  devServer: {

    contentBase: './src/main/resources/templates/',

    proxy: {  
     '/api/*': {
       target: 'http://localhost:8080',
       secure: false,
       publicPath: "/"
     
      },
    },
    
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