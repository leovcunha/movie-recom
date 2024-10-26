import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom";
import { discoverMovies } from "./actions/discoverMovies";
import Header from "./components/Header";
import MovieTable from "./components/MovieTable";
import "./css/main.css";

const App = () => {
    const [isLoading, setIsLoading] = useState(false);
    const [moviePage, setMoviePage] = useState(1);
    const [movieList, setMovieList] = useState({ results: [] });
    const [error, setError] = useState(null);

    useEffect(() => {
        triggerMoviesUpdate();
    }, [moviePage]);

    const triggerMoviesUpdate = async () => {
        try {
            setIsLoading(true);
            const res = await discoverMovies(moviePage.toString());
            setMovieList(res.data);
        } catch (err) {
            setError("Failed to fetch movies");
            console.error("Error:", err);
        } finally {
            setIsLoading(false);
        }
    };

    const pageHandler = (increment) => {
        console.log("App pageHandler called with increment:", increment);
        console.log("Current page before update:", moviePage);

        setMoviePage((prev) => {
            const newPage = increment ? prev + 1 : Math.max(1, prev - 1);
            console.log("New page will be:", newPage);
            return newPage;
        });
    };

    return (
        <div className="app-container">
            <Header />
            {error && <div className="error">{error}</div>}
            {isLoading ? (
                <div>Loading...</div>
            ) : (
                <MovieTable
                    movies={movieList.results}
                    pagehandler={(increment) => {
                        console.log("MovieTable triggered pageHandler with:", increment);
                        pageHandler(increment);
                    }}
                    currentPage={moviePage}
                />
            )}
        </div>
    );
};

ReactDOM.render(<App />, document.getElementById("react"));

export default App;
