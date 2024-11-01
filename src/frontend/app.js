import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { discoverMovies } from './actions/discoverMovies';
import { fetchPopularMovies } from './actions/fetchPopularMovies';
import Header from './components/Header';
import MovieTable from './components/MovieTable';
import '../main/main.css';

const App = () => {
    const [isLoading, setIsLoading] = useState(false);
    const [moviePage, setMoviePage] = useState(1);
    const [movieList, setMovieList] = useState({ results: [] });
    const [popularMovies, setPopularMovies] = useState({ results: [] });
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchInitialMovies();
    }, []);

    useEffect(() => {
        triggerMoviesUpdate();
    }, [moviePage]);

    const fetchInitialMovies = async () => {
        try {
            setIsLoading(true);
            const res = await fetchPopularMovies();
            setPopularMovies(res.data);
            // After loading popular movies, fetch the regular discover movies
            const discoverRes = await discoverMovies(moviePage.toString());
            setMovieList(discoverRes.data);
        } catch (err) {
            setError('Failed to fetch movies');
            console.error('Error:', err);
        } finally {
            setIsLoading(false);
        }
    };

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
                <>
                    {popularMovies.results.length > 0 && (
                        <div className="popular-movies">
                            <h2>Most Viewed Movies</h2>
                            <MovieTable
                                movies={popularMovies.results}
                                pagehandler={() => {}}
                                currentPage={1}
                            />
                        </div>
                    )}
                    <h2>Discover Movies</h2>
                    <MovieTable
                        movies={movieList.results}
                        pagehandler={(increment) => pageHandler(increment)}
                        currentPage={moviePage}
                    />
                </>
            )}
        </div>
    );
};

const container = document.getElementById('react');
const root = createRoot(container);
root.render(<App />);

export default App;
