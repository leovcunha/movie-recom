const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
const ReactRefreshWebpackPlugin = require('@pmmmwh/react-refresh-webpack-plugin');
const TerserPlugin = require('terser-webpack-plugin');

module.exports = (env, argv) => {
    const isProduction = argv.mode === 'production';

    return {
        mode: isProduction ? 'production' : 'development',
        entry: './src/frontend/app.js',
        output: {
            path: path.resolve(__dirname, 'src/main/resources/static/'),
            filename: isProduction ? 'built/[name].[contenthash].js' : 'built/[name].bundle.js',
            chunkFilename: isProduction
                ? 'built/[name].[contenthash].chunk.js'
                : 'built/[name].chunk.js',
            clean: true,
        },
        optimization: {
            runtimeChunk: 'single',
            splitChunks: {
                chunks: 'all',
                maxInitialRequests: Infinity,
                minSize: 0,
                cacheGroups: {
                    vendor: {
                        test: /[\\/]node_modules[\\/]/,
                        name(module) {
                            const packageName = module.context.match(
                                /[\\/]node_modules[\\/](.*?)([\\/]|$)/
                            )[1];
                            return `vendor.${packageName.replace('@', '')}`;
                        },
                    },
                },
            },
            minimize: isProduction,
            minimizer: [new TerserPlugin(), new CssMinimizerPlugin()],
        },
        module: {
            rules: [
                {
                    test: /\.(js|jsx)$/,
                    exclude: /node_modules/,
                    use: {
                        loader: 'babel-loader',
                        options: {
                            presets: ['@babel/preset-env', '@babel/preset-react'],
                            plugins: [!isProduction && 'react-refresh/babel'].filter(Boolean),
                        },
                    },
                },
                {
                    test: /\.css$/,
                    use: [
                        isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
                        {
                            loader: 'css-loader',
                            options: {
                                sourceMap: !isProduction,
                            },
                        },
                    ],
                },
            ],
        },
        plugins: [
            new webpack.DefinePlugin({
                'process.env.BACKEND_URL': JSON.stringify(
                    process.env.BACKEND_URL || 'http://localhost:8000'
                ),
            }),
            new HtmlWebpackPlugin({
                template: './src/main/index.html',
                filename: 'index.html',
            }),
            ...(!isProduction ? [new ReactRefreshWebpackPlugin()] : []),
            isProduction &&
                new MiniCssExtractPlugin({
                    filename: 'built/styles/[name].[contenthash].css',
                }),
        ].filter(Boolean),
        resolve: {
            extensions: ['.js', '.jsx'],
            fallback: {
                path: require.resolve('path-browserify'),
            },
        },
        devServer: {
            static: {
                directory: path.join(__dirname, 'src/main/resources/static/'),
            },
            historyApiFallback: true,
            hot: !isProduction,
            proxy: {
                '/api/*': {
                    target: 'http://localhost:8000',
                    secure: false,
                },
            },
        },
    };
};
