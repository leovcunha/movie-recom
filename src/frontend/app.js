import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { discoverMovies } from './actions/discoverMovies';
import Header from './components/Header';
import MovieTable from './components/MovieTable';
import '../main/main.css';

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
            setError('Failed to fetch movies');
            console.error('Error:', err);
        } finally {
            setIsLoading(false);
        }
    };

    const pageHandler = (increment) => {
        setMoviePage((prev) => {
            const newPage = increment ? prev + 1 : Math.max(1, prev - 1);
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
                        pageHandler(increment);
                    }}
                    currentPage={moviePage}
                />
            )}
        </div>
    );
};

const container = document.getElementById('react');
const root = createRoot(container);
root.render(<App />);

export default App;
