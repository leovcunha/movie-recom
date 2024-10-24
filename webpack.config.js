const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");

module.exports = {
    entry: "./src/frontend/app.js",

    output: {
        path: path.resolve(__dirname, "src/main/resources/static/"),
        filename: "built/app-bundle.js",
        publicPath: "/",
    },

    plugins: [
        new HtmlWebpackPlugin({
            hash: true,
            template: "src/main/index.html",
            filename: "index.html",
        }),
    ],

    devServer: {
        static: {
            directory: path.join(__dirname, "src/main/resources/static/"),
        },
        proxy: {
            "/api/*": {
                target: "http://localhost:8080",
                secure: false,
                publicPath: "/",
            },
        },
    },
    module: {
        rules: [
            {
                test: /\.js$/,
                exclude: /(node_modules|bower_components)/,
                use: {
                    loader: "babel-loader",
                },
            },
            {
                test: /\.css$/,
                use: ["style-loader", "css-loader"],
            },
        ],
    },
};
